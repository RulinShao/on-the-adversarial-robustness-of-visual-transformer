from utils import *
from auto_LiRPA.utils import MultiAverageMeter
import vit_jax.hyper as hyper

def evaluate_natural(args, model, test_loader, verbose=False):
    model.eval()

    with torch.no_grad():
        meter = MultiAverageMeter()
        test_loss = test_acc = test_n = 0

        def test_step(step, X_batch, y_batch):
            X, y = X_batch.cuda(), y_batch.cuda()
            if args.method == 'natural':
                output = model(X)
                loss = F.cross_entropy(output, y)
            else:
                raise NotImplementedError(args.method)
            meter.update('test_loss', loss.item(), y.size(0))
            meter.update('test_acc', (output.max(1)[1] == y).float().mean(), y.size(0))
            if step % args.log_interval == 0 and verbose:
                logger.info('Eval step {}/{} {}'.format(
                    step, total_test_steps, meter))               

        for step, (X_batch, y_batch) in enumerate(test_loader):
            test_step(step, X_batch, y_batch)

        logger.info('Evaluation {}'.format(meter))  


def train_natural(args, model):
    if args.optimizer == 'sgd':
        opt = torch.optim.SGD(model.parameters(), lr=args.base_lr, momentum=0.9)
    elif args.optimizer == 'adam':
        opt = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    else:
        raise ValueError(args.optimizer)

    start_epoch_time = time.time()
    meter = MultiAverageMeter()

    if args.data_loader == 'torch':
        train_loader, test_loader = get_loaders(args)
    else:
        test_loader = ds_test

    def train_step(step, X_batch, y_batch, lr_repl):
        lr = float(lr_repl[0])
        opt.param_groups[0]['lr'] = lr
        model.train()

        batch_size = math.ceil(args.batch_size / args.accum_steps)
        for i in range(args.accum_steps):
            X = X_batch[i*batch_size:(i+1)*batch_size].cuda()
            y = y_batch[i*batch_size:(i+1)*batch_size].cuda()

            if args.method == 'natural':
                output = model(X)
                loss = F.cross_entropy(output, y)
                (loss*(X.shape[0]/X_batch.shape[0])).backward() 
            else:
                raise NotImplementedError(args.method)

            meter.update('train_loss', loss.item(), y.size(0))
            meter.update('train_acc', (output.max(1)[1] == y).float().mean(), y.size(0))

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        opt.step()
        opt.zero_grad()

        if step % args.log_interval == 0:
            logger.info('Training step {} lr {:.4f} {}'.format(
                step, lr, meter))   
            train_loss = 0
            train_acc = 0
            train_n = 0

        if step % args.save_interval == 0:
            path = os.path.join(args.out_dir, 'checkpoint_{}'.format(step))
            torch.save({ 'state_dict': model.state_dict(), 'step': step }, path)
            logger.info('Checkpoint saved to {}'.format(path))

        if step == total_steps:
            return

    total_steps = args.epochs * len(train_loader) 
    lr_fn = hyper.create_learning_rate_schedule(total_steps, args.base_lr, 'cosine', args.warmup_steps)
    lr_iter = hyper.lr_prefetch_iter(lr_fn, 0, total_steps)

    step = 0
    for epoch in range(1, args.epochs+1):
        logger.info('Epoch {}'.format(epoch))
        for (X_batch, y_batch) in train_loader:
            step += 1
            lr_repl = next(lr_iter)
            train_step(step, X_batch, y_batch, lr_repl)
        evaluate_natural(args, model, test_loader)            
