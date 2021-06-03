
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from autoattack import AutoAttack
from utils import normalize
# installing AutoAttack by: pip install git+https://github.com/fra31/auto-attack

def evaluate_aa(args, model):
    test_transform_nonorm = transforms.Compose([
        transforms.ToTensor()
    ])
    num_workers = 2
    test_dataset_nonorm = datasets.CIFAR10(
        args.data_dir, train=False, transform=test_transform_nonorm, download=True)
    test_loader_nonorm = torch.utils.data.DataLoader(
        dataset=test_dataset_nonorm,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=2,
    )

    model.eval()
    l = [x for (x, y) in test_loader_nonorm]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader_nonorm]
    y_test = torch.cat(l, 0)
    class normalize_model():
        def __init__(self, model):
            self.model_test = model
        def __call__(self, x):
            x_norm = normalize(x)
            return self.model_test(x_norm)
    new_model = normalize_model(model)
    epsilon = 8 / 255.
    adversary = AutoAttack(new_model, norm='Linf', eps=epsilon, version='standard')
    X_adv = adversary.run_standard_evaluation(x_test, y_test, bs=128)
