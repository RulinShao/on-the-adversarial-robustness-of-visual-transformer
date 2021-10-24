import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torchvision
import numpy as np
import os
import random
import argparse

from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, L2CarliniWagnerAttack
from autoattack import AutoAttack
import eagerpy as ep
from timm.models import load_checkpoint, create_model
import torch_dct as dct

import utils as aa


parser = argparse.ArgumentParser(description='On the Adversarial Robustness of Visual Transformer')
parser.add_argument('--data_dir', help='path to ImageNet dataset')
parser.add_argument('--num-classes', type=int, default=1000, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--seed', default=310)
parser.add_argument('--attack_batch_size', default=40)
parser.add_argument('--attack_epochs', default=25)
parser.add_argument('--attack_type', default='LinfPGD', help="attack type for foolbox attack")
parser.add_argument('--iteration', default=40)
parser.add_argument('--model', default='vit_small_patch16_224', type=str)
parser.add_argument('--mode', default='foolbox')

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Put your checkpoint path here
checkpoint_paths = {}


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_acc(model, inputs, labels):
    with torch.no_grad():
        predictions = model(inputs).argmax(axis=-1)
        accuracy = (predictions == labels).float().mean()
        return accuracy.item()


def count_parameters():
    model = get_model()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model}: {count}")
    return count


def get_model(model_name=None):
    # load pre-trained models
    if not model_name:
        model_name = args.model
    if model_name == 'resnet18':
        model = torchvision.models.resnet18(pretrained=True)
    elif model_name == 'alexnet':
        model = torchvision.models.alexnet(pretrained=True)
    elif model_name == 'squeezenet':
        model = torchvision.models.squeezenet1_0(pretrained=True)
    elif model_name == 'vgg16':
        model = torchvision.models.vgg16(pretrained=True)
    elif model_name == 'densenet':
        model = torchvision.models.densenet161(pretrained=True)
    elif model_name == 'inception':
        model = torchvision.models.inception_v3(pretrained=True)
    elif model_name == 'googlenet':
        model = torchvision.models.googlenet(pretrained=True)
    elif model_name == 'shufflenet':
        model = torchvision.models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'mobilenet':
        model = torchvision.models.mobilenet_v2(pretrained=True)
    elif model_name == 'resnet50_32x4d':
        model = torchvision.models.resnext50_32x4d(pretrained=True)
    elif model_name == 'wide_resnet50_2':
        model = torchvision.models.wide_resnet50_2(pretrained=True)
    elif model_name == 'mnasnet':
        model = torchvision.models.mnasnet1_0(pretrained=True)
    elif model_name == 'resnext50_32x4d_ssl':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_ssl')
    elif model_name == 'resnext50_32x4d_swsl':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
    elif model_name == 'resnet50_swsl':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')
    else:
        model = create_model(model_name,
                             pretrained=True,
                             num_classes=args.num_classes,
                             in_chans=3,)
    return model.eval().to(device)


def get_val_loader(batch_size=None, input_size=224, normalize=False, model_name=None):

    if args.seed:
        seed_everything(args.seed)

    if not batch_size:
        batch_size = args.attack_batch_size

    if '384' in args.model or (model_name and '384' in model_name):
        input_size = 384

    if normalize:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        valdir = os.path.join(args.data_dir, 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(input_size + 32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True)
    else:
        valdir = os.path.join(args.data_dir, 'val')
        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(input_size+32),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
            ])),
            batch_size=batch_size, shuffle=True,
            num_workers=2, pin_memory=True)

    return val_loader


def evaluate(model=None, model_name=None):

    if not model:
        model = get_model()

    val_loader = get_val_loader(normalize=True, model_name=model_name)
    criterion = nn.CrossEntropyLoss()

    val_accuracy, val_loss = 0.0, 0.0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            acc = (outputs.argmax(axis=-1) == labels).float().mean()

            val_accuracy += acc / len(val_loader)
            val_loss += loss / len(val_loader)

    print("Model: {}, Loss: {}, accuracy: {}".format(model_name or args.model, val_loss, val_accuracy))


def foolbox_attack(filter=None, filter_preserve='low', free_parm='eps', plot_num=None):
    # get model.
    model = get_model()
    model = nn.DataParallel(model).to(device)
    model = model.eval()

    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    if plot_num:
        free_parm = ''
        val_loader = get_val_loader(plot_num)
    else:
        # Load images.
        val_loader = get_val_loader(args.attack_batch_size)

    if 'eps' in free_parm:
        epsilons = [0.001, 0.003, 0.005, 0.008, 0.01, 0.1]
    else:
        epsilons = [0.01]
    if 'step' in free_parm:
        steps = [1, 5, 10, 30, 40, 50]
    else:
        steps = [args.iteration]

    for step in steps:
        # Adversarial attack.
        if args.attack_type == 'LinfPGD':
            attack = LinfPGD(steps=step)
        elif args.attack_type == 'FGSM':
            attack = FGSM()

        clean_acc = 0.0

        for i, data in enumerate(val_loader, 0):

            # Samples (attack_batch_size * attack_epochs) images for adversarial attack.
            if i >= args.attack_epochs:
                break

            images, labels = data[0].to(device), data[1].to(device)
            if step == steps[0]:
                clean_acc += (get_acc(fmodel, images, labels)) / args.attack_epochs  # accumulate for attack epochs.

            _images, _labels = ep.astensors(images, labels)
            raw_advs, clipped_advs, success = attack(fmodel, _images, _labels, epsilons=epsilons)

            if plot_num:
                grad = torch.from_numpy(raw_advs[0].numpy()).to(device) - images
                grad = grad.clone().detach_()
                return grad

            if filter:
                robust_accuracy = torch.empty(len(epsilons))
                for eps_id in range(len(epsilons)):
                    grad = torch.from_numpy(raw_advs[eps_id].numpy()).to(device) - images
                    grad = grad.clone().detach_()
                    freq = dct.dct_2d(grad)
                    if filter_preserve == 'low':
                        mask = torch.zeros(freq.size()).to(device)
                        mask[:, :, :filter, :filter] = 1
                    elif filter_preserve == 'high':
                        mask = torch.zeros(freq.size()).to(device)
                        mask[:, :, filter:, filter:] = 1
                    masked_freq = torch.mul(freq, mask)
                    new_grad = dct.idct_2d(masked_freq)
                    x_adv = torch.clamp(images + new_grad, 0, 1).detach_()

                    robust_accuracy[eps_id] = (get_acc(fmodel, x_adv, labels))
            else:
                robust_accuracy = 1 - success.float32().mean(axis=-1)
            if i == 0:
                robust_acc = robust_accuracy / args.attack_epochs
            else:
                robust_acc += robust_accuracy / args.attack_epochs

        if step == steps[0]:
            print("sample size is : ", args.attack_batch_size * args.attack_epochs)
            print(f"clean accuracy:  {clean_acc * 100:.1f} %")
            print(f"Model {args.model} robust accuracy for {args.attack_type} perturbations with")
        for eps, acc in zip(epsilons, robust_acc):
            print(f"  Step {step}, Linf norm ≤ {eps:<6}: {acc.item() * 100:4.1f} %")
        print('  -------------------')


def auto_attack():
    # get model.
    model = get_model()
    model = nn.Sequential(aa.get_normalize_layer('imagenet'), model)
    model = nn.DataParallel(model).to(device)
    model = model.eval()

    # Load images.
    val_loader = get_val_loader(args.attack_batch_size)

    # Adversarial attack.
    attack = AutoAttack

    epsilons = [0.001, 0.003, 0.005, 0.008, 0.01, 0.1]
    clean_acc = 0.0
    robust_acc = [0.0] * len(epsilons)

    for i, data in enumerate(val_loader, 0):

        # Samples (attack_batch_size * attack_epochs) images for adversarial attack.
        if i >= args.attack_epochs:
            break

        images, labels = data[0].to(device), data[1].to(device)

        clean_acc += (get_acc(model, images, labels)) / args.attack_epochs

        for j in range(len(epsilons)):
            adversary = attack(model, norm='Linf', eps=epsilons[j])
            x_adv = adversary.run_standard_evaluation(images, labels)
            adv_acc = get_acc(model, x_adv, labels)
            robust_acc[j] += adv_acc / args.attack_epochs

    print("sample size is : ", args.attack_batch_size * args.attack_epochs)
    print(f"clean accuracy:  {clean_acc * 100:.1f} %")
    print(f"Model {args.model} robust accuracy for AutoAttack perturbations with")
    for eps, acc in zip(epsilons, robust_acc):
        print(f"  Linf norm ≤ {eps:<6}: {acc * 100:4.1f} %")


if __name__ == "__main__":
    if args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'foolbox':
        foolbox_attack()
    elif args.mode == 'auto':
        auto_attack()
    elif args.mode == 'count':
        count_parameters()
    elif args.mode == 'foolbox-filter':
        foolbox_attack(filter=32, filter_preserve='high')
    elif args.mode == 'foolbox-eps-step':
        foolbox_attack(free_parm='eps-step')
    else:
        print("Unknown mode!")
