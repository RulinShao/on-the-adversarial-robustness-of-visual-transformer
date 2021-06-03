from utils import *
from pgd import evaluate_pgd
from torch.autograd import Variable
import pdb

def train_adv(args, model, ds_train, ds_test):
    assert args.data_loader == 'torch'

    mu = torch.tensor(cifar10_mean).view(3,1,1).cuda()
    std = torch.tensor(cifar10_std).view(3,1,1).cuda()

    upper_limit = ((1 - mu)/ std)
    lower_limit = ((0 - mu)/ std)

    epsilon = (args.epsilon / 255.) / std
    alpha = (args.alpha / 255.) / std
    pgd_alpha = (2 / 255.) / std   

    train_loader, test_loader = ds_train, ds_test

    criterion = nn.CrossEntropyLoss()

    steps_per_epoch = len(train_loader)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.delta_init == 'previous':
        delta = torch.zeros(args.batch_size, 3, 32, 32).cuda()
    lr_steps = args.epochs * steps_per_epoch
    if args.lr_schedule == 'cyclic':
        scheduler = torch.optim.lr_scheduler.CyclicLR(opt, base_lr=args.lr_min, max_lr=args.lr_max,
            step_size_up=lr_steps / 2, step_size_down=lr_steps / 2)
    elif args.lr_schedule == 'multistep':
        milestones = list(np.array(args.lr_decay_milestones)*steps_per_epoch)
        logger.info('LR milestones {} {}'.format(args.lr_decay_milestones, milestones))
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)

    if args.load and not args.load_state_dict_only:
        checkpoint = torch.load(args.load)
        epoch_s = checkpoint['epoch']
        if 'opt' in checkpoint:
            opt.load_state_dict(checkpoint['opt'])  
        else:
            logger.warning('No state_dict for optimizer in the checkpoint')
        logger.info('Checkpoint at epoch {}'.format(epoch_s))
        for i in range(epoch_s):
            for j in range(steps_per_epoch):
                scheduler.step()        
    else:
        epoch_s = 0

    def evaluate():
        eval_steps = 1
        pgd_loss, pgd_acc = evaluate_pgd(args, model, test_loader, eval_steps=eval_steps)
        logger.info('PGD ({} samples): loss {:.4f} acc {:.4f}'.format(eval_steps*args.batch_size_eval, pgd_loss, pgd_acc))
        opt.zero_grad()

    evaluate()

    prev_robust_acc = 0.
    start_train_time = time.time()
    for epoch in range(epoch_s+1, args.epochs+1):
        start_epoch_time = time.time()
        train_loss = 0
        train_acc = 0
        train_n = 0
    
        def train_step(X, y, alpha):
            model.train()

            if args.method == 'pgd':
                alpha = pgd_alpha
                delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    for i in range(len(epsilon)):
                        delta[:, i, :, :].uniform_(-epsilon[i][0][0].item(), epsilon[i][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                for _ in range(args.attack_iters):
                    output = model(X + delta)
                    loss = criterion(output, y)
                    grad = torch.autograd.grad(loss, delta)[0].detach()
                    delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta = delta.detach()
                output = model(X + delta)
                loss = criterion(output, y)
            elif args.method == 'trades':
                beta = 6.0

                batch_size = len(X)

                delta = torch.zeros_like(X).cuda()

                for i in range(len(epsilon)):
                    delta[:, i, :, :].uniform_(-0.001/cifar10_std[i], 0.001/cifar10_std[i])
                delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                x_adv = X.detach() + delta.detach()

                criterion_kl = nn.KLDivLoss(size_average=False)
                model.eval()

                for _ in range(args.attack_iters):
                    x_adv.requires_grad_()
                    loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                        F.softmax(model(X), dim=1))
                    grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                    with torch.no_grad():
                        x_adv = x_adv.detach() + pgd_alpha * torch.sign(grad.detach())
                        x_adv = torch.min(torch.max(x_adv, X - epsilon), X + epsilon)
                        x_adv = torch.min(torch.max(x_adv, lower_limit), upper_limit)

                model.train()

                x_adv = Variable(x_adv, requires_grad=False)

                output = logits = model(X)
                loss_natural = F.cross_entropy(logits, y)
                loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(model(x_adv), dim=1),
                    F.softmax(model(X), dim=1))
                loss = loss_natural + beta * loss_robust

            elif args.method == 'fgsm':
                if args.delta_init != 'previous':
                    delta = torch.zeros_like(X).cuda()
                if args.delta_init == 'random':
                    for j in range(len(epsilon)):
                        delta[:, j, :, :].uniform_(-epsilon[j][0][0].item(), epsilon[j][0][0].item())
                    delta.data = clamp(delta, lower_limit - X, upper_limit - X)
                delta.requires_grad = True
                output = model(X + delta[:X.size(0)])
                loss = F.cross_entropy(output, y)
                grad = torch.autograd.grad(loss, delta)[0].detach()
                # loss.backward()
                # grad = delta.grad.detach()
                delta.data = clamp(delta + alpha * torch.sign(grad), -epsilon, epsilon)
                delta.data[:X.size(0)] = clamp(delta[:X.size(0)], lower_limit - X, upper_limit - X)
                delta = delta.detach()
                output = model(X + delta[:X.size(0)])
                loss = criterion(output, y)
            else:
                raise ValueError(args.method)

            (loss/args.accum_steps).backward()
            acc = (output.max(1)[1] == y).float().mean()
            
            return loss, acc
        
        for step, (X, y) in enumerate(train_loader):
            batch_size = args.batch_size // args.accum_steps
            for t in range(args.accum_steps):
                X_ = X[t*batch_size:(t+1)*batch_size].cuda()#.permute(0, 3, 1, 2)
                y_ = y[t*batch_size:(t+1)*batch_size].cuda()#.max(dim=-1).indices
                if len(X_) == 0:
                    break
                loss, acc = train_step(X_, y_, alpha)
                train_loss += loss.item() * y_.size(0)
                train_acc += acc.item() * y_.size(0)
                train_n += y_.size(0)

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            opt.step()
            opt.zero_grad()

            if (step+1) % args.log_interval == 0 or step+1 == steps_per_epoch:
                logger.info('Training epoch {} step {}/{}, lr {:.4f} loss {:.4f} acc {:.4f}'.format(
                    epoch, step+1, len(train_loader), 
                    opt.param_groups[0]['lr'], 
                    train_loss/train_n, train_acc/train_n
                ))     

            scheduler.step()

        evaluate()

        path = os.path.join(args.out_dir, 'checkpoint_{}'.format(epoch))
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch, 'opt': opt.state_dict()}, path)
        logger.info('Checkpoint saved to {}'.format(path))
