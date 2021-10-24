import torch
import torch.nn as nn
import argparse
from foolbox import PyTorchModel, accuracy, samples
from foolbox.attacks import LinfPGD, FGSM, L2CarliniWagnerAttack
import eagerpy as ep

from white_box_test import get_model, get_val_loader, get_acc, count_parameters


parser = argparse.ArgumentParser(description='Transferability test')
parser.add_argument('--data_dir', help='path to ImageNet dataset')
parser.add_argument('--seed', default=310)
parser.add_argument('--attack_batch_size', default=100)
parser.add_argument('--attack_epochs', default=10)
parser.add_argument('--attack_type', default='FGSM')
parser.add_argument('--epsilon', default=0.1)
parser.add_argument('--iteration', default=1)

args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def transfer_attack():
    model_name = ['vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224',
                  'resnext50_32x4d_ssl', 'resnet50_swsl', 'resnet50_32x4d', 'mobilenet', 'vgg16', 'resnet18', 'shufflenet']
    epsilon_list = [0.1, 0.08, 0.05, 0.02, 0.01, 0.005]
    for eps in epsilon_list:
        transfer_asr_list = []

        for i in range(len(model_name)):  # Index of the fmodel.
            transfer_asr_sublist = []
            for j in range(len(model_name)):  # Index of the eval_model.

                model_1, model_2 = get_model(model_name[i]), get_model(model_name[j])
                model_1, model_2 = nn.DataParallel(model_1).to(device).eval(), nn.DataParallel(model_2).to(device).eval()

                preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
                fmodel = PyTorchModel(model_1, bounds=(0, 1), preprocessing=preprocessing)
                eval_model = PyTorchModel(model_2, bounds=(0, 1), preprocessing=preprocessing)

                # Load images.
                val_loader = get_val_loader(args.attack_batch_size, model_name=model_name)

                # Adversarial attack.
                clean_acc_1, clean_acc_2 = 0.0, 0.0

                epsilons = [eps]

                if args.attack_type == 'LinfPGD':
                    attack = LinfPGD(steps=args.iteration)
                elif args.attack_type == 'FGSM':
                    attack = FGSM()
                elif args.attack_type == 'CW':
                    attack = L2CarliniWagnerAttack(steps=args.iteration)

                for iter, data in enumerate(val_loader, 0):

                    # Samples (attack_batch_size * attack_epochs) images for adversarial attack.
                    if iter >= args.attack_epochs:
                        break

                    images, labels = data[0].to(device), data[1].to(device)
                    _images, _labels = ep.astensors(images, labels)

                    clean_acc_1 += (get_acc(fmodel, images, labels)) / args.attack_epochs  # accumulate for attack epochs.
                    clean_acc_2 += (get_acc(eval_model, images, labels)) / args.attack_epochs

                    raw_advs, clipped_advs, success = attack(fmodel, _images, _labels, epsilons=epsilons)
                    clipped_advs = torch.from_numpy(clipped_advs[0].numpy()).to(device)

                    if iter == 0:
                        attack_acc = (1 - success.float32().mean(axis=-1)) / args.attack_epochs
                        transfer_acc = (get_acc(eval_model, clipped_advs, labels)) / args.attack_epochs
                    else:
                        attack_acc += (1 - success.float32().mean(axis=-1)) / args.attack_epochs
                        transfer_acc += (get_acc(eval_model, clipped_advs, labels)) / args.attack_epochs

                if (i + j) == 0:
                    if args.attack_type == 'FGSM':
                        print(f"\n Attack Type: {args.attack_type}, epsilon: {epsilons[0]}"
                              f", sample size: {args.attack_batch_size * args.attack_epochs} \n")
                    else:
                        print(f"\n Attack Type: {args.attack_type}, epsilon: {epsilons[0]}, iteration:{args.iteration}"
                              f", sample size: {args.attack_batch_size * args.attack_epochs} \n")

                print(
                    f"{model_name[i]} --> {model_name[j]}: ({clean_acc_1 * 100} % --> {attack_acc * 100} %) "
                    f"----> ({clean_acc_2 * 100} % --> {transfer_acc * 100} %) --- {(clean_acc_2-transfer_acc)*100} % dropped")
                print(f"\n{model_name[i]} --> {model_name[j]}: ")
                print(f"  clean_acc1: {clean_acc_1 * 100} %")
                print(f"  clean_acc2: {clean_acc_2 * 100} %")
                print(f"  white_attack_acc: {attack_acc * 100} %")
                print(f"  transfer_attack_acc: {transfer_acc * 100} %")
                transfer_asr_sublist.append((1-transfer_acc)*100)
            transfer_asr_list.append(transfer_asr_sublist)
        print(f"eps: {eps}")
        print(transfer_asr_list)

        import pickle
        with open(f'transfer_eps_{int(eps*1000)}.pkl', 'wb') as f:
            pickle.dump(transfer_asr_list, f)


if __name__ == "__main__":
    transfer_attack()