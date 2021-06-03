import argparse
import logging
import timm
# import apex.amp as amp
from parser import get_args

from vit_jax import input_pipeline
from vit_jax.input_pipeline import DATASET_PRESETS 
from utils import *
from train_adv import train_adv
from pgd import evaluate_pgd
from natural import train_natural, evaluate_natural
from preact_resnet import PreActResNet18
from wideresnet import WideResNet
from evaluate import evaluate_aa


def main():
    args = get_args()

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    logfile = os.path.join(args.out_dir, 'output.log')

    file_handler = logging.FileHandler(logfile)
    file_handler.setFormatter(logging.Formatter('%(levelname)-8s %(asctime)-12s %(message)s'))
    logger.addHandler(file_handler)      

    logger.info(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    resize_size = args.resize
    crop_size = args.crop

    assert args.data_loader == 'torch'
    train_loader, test_loader = get_loaders(args)

    if 'vit' in args.model:
        num_classes = args.num_classes
        from timm_vit.vit import (
            vit_base_patch2, vit_base_patch16_224_in21k, vit_large_patch16_224_in21k)
        model = eval(args.model)(
            pretrained=(not args.scratch), 
            img_size=crop_size, num_classes=num_classes, patch_size=args.patch, args=args).cuda()
        logger.info('Model {}'.format(model))
    else:
        model = eval(args.model)(in_dim=crop_size).cuda()
    model.train()

    if args.load:
        checkpoint = torch.load(args.load)
        model.load_state_dict(checkpoint['state_dict'])

    if args.method == 'natural':
        if not args.eval:
            ds_train = input_pipeline.get_data(
                dataset=args.data, mode='train',
                repeats=None, mixup_alpha=0, batch_size=args.batch_size, shuffle_buffer=50_000,
                tfds_data_dir=args.tfds_dir, tfds_manual_dir=args.tfds_dir, resize_size=resize_size,
                crop_size=crop_size,
                inception_crop=(not args.no_inception_crop))
            logger.info('VIT ds_train {}'.format(ds_train))
        ds_test = input_pipeline.get_data(
            dataset=args.data, mode='test',
            repeats=1, batch_size=args.batch_size_eval, tfds_data_dir=args.tfds_dir, 
            tfds_manual_dir=args.tfds_dir, crop_size=crop_size)
        logger.info('VIT ds_test {}'.format(ds_test))        
        if args.eval:
            evaluate_natural(args, model, ds_test, verbose=True)
        else:
            train_natural(args, model, ds_train, ds_test)
    elif args.method in ['fgsm', 'pgd', 'trades']:
        if args.eval_all:
            acc = []
            for i in range(1, args.epochs+1):
                logger.info('Evaluating epoch {}'.format(i))
                checkpoint = torch.load(os.path.join(args.out_dir, 'checkpoint_{}'.format(i)))
                model.load_state_dict(checkpoint['state_dict'])
                loss_, acc_ = evaluate_pgd(args, model, test_loader)
                logger.info('Acc: {:.5f}'.format(acc_))
                acc.append(acc_)
            print(acc)
        elif args.eval_aa:
            evaluate_aa(args, model)
        elif args.eval:
            evaluate_pgd(args, model, test_loader)
        else:
            train_adv(args, model, train_loader, test_loader)
    else:
        raise ValueError(args.method)

if __name__ == "__main__":
    main()
