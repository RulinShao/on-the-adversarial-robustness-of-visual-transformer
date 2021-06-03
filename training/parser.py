import argparse

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='vit_base_patch32_384')
    parser.add_argument('--method', type=str, default='natural',
                        choices=['natural', 'fgsm', 'pgd', 'trades'])
    parser.add_argument('--run-dummy', action='store_true')
    parser.add_argument('--accum-steps', type=int, default=1)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--save-interval', type=int, default=1000)
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument('--prefetch', type=int, default=1)#2)
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--tfds-dir', type=str, default='~/dataset/tar')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval-steps', type=int, default=1000)
    parser.add_argument('--batch-size', default=128, type=int)
    parser.add_argument('--batch-size-eval', default=512, type=int)
    parser.add_argument('--eval-interval', type=int, default=100)
    parser.add_argument('--no-timm', action='store_true')
    parser.add_argument('--crop', type=int, default=32)
    parser.add_argument('--resize', type=int, default=32)
    parser.add_argument('--load', type=str)
    parser.add_argument('--data-loader', type=str, default='torch', choices=['torch'])
    parser.add_argument('--no-inception-crop', action='store_true')
    parser.add_argument('--scratch', action='store_true')
    parser.add_argument('--custom-vit', action='store_true')
    parser.add_argument('--base-lr', type=float, default=0.03)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--depth', type=int, default=12)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--attack-iters', type=int, default=7, help='for pgd training')
    parser.add_argument('--patch', type=int, default=4)
    parser.add_argument('--load-state-dict-only', action='store_true')
    parser.add_argument('--patch-embed-scratch', action='store_true')
    parser.add_argument('--num-layers', type=int)

    parser.add_argument('--eval-restarts', type=int, default=1)
    parser.add_argument('--eval-iters', type=int, default=10)
    parser.add_argument('--downsample-factor', action='store_true')
    parser.add_argument('--eval-all', action='store_true')
    parser.add_argument('--eval-aa', action='store_true')
    parser.add_argument('--num-classes', type=int, default=10)

    parser.add_argument('--data-dir', default='./data', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--lr-decay-milestones', type=int, nargs='+', default=[15,18])
    parser.add_argument('--lr-schedule', default='multistep', choices=['cyclic', 'multistep'])
    parser.add_argument('--lr-min', default=0., type=float)
    parser.add_argument('--lr-max', default=0.1, type=float)
    parser.add_argument('--lr-natural', default=5e-4, type=float)
    parser.add_argument('--weight-decay', default=2e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--epsilon', default=8, type=int)
    parser.add_argument('--alpha', default=10, type=float, help='Step size')
    parser.add_argument('--delta-init', default='random', choices=['zero', 'random', 'previous'],
        help='Perturbation initialization method')
    parser.add_argument('--out-dir', '--dir', default='output_dir', type=str, help='Output directory')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--early-stop', action='store_true', help='Early stop if overfitting occurs')
    # parser.add_argument('--opt-level', default='O2', type=str, choices=['O0', 'O1', 'O2'],
    #     help='O0 is FP32 training, O1 is Mixed Precision, and O2 is "Almost FP16" Mixed Precision')
    # parser.add_argument('--loss-scale', default='1.0', type=str, choices=['1.0', 'dynamic'],
    #     help='If loss_scale is "dynamic", adaptively adjust the loss scale over time')
    # parser.add_argument('--master-weights', action='store_true',
    #     help='Maintain FP32 master weights to accompany any FP16 model weights, not applicable for O1 opt level')

    parser.add_argument('--pretrain-pos-only', action='store_true')

    args = parser.parse_args()
    assert args.batch_size % args.accum_steps == 0

    return args