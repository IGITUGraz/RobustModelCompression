import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Training")

    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        help='model architecture (default: resnet50)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--test_all', action='store_true',
                        help='Run all validation set (default: will run 5k test samples from RobustBench)')
    parser.add_argument('-c', '--corruption', action='store_true',
                        help='Corruptions to be used for test set evaluations')
    parser.add_argument('--augmentation', default="standard", type=str,
                        help='augmentation type')
    parser.add_argument('--dir', default='/data/', type=str, metavar='DIR',
                        help='Path to dataset ')
    parser.add_argument('--mixing_set', default='/data/fractals_and_fvis/', type=str, metavar='DIR',
                        help='Path to mixing set')
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='Evaluation mini-batch size (default: 1)')
    parser.add_argument('--path', default=None, type=str, metavar='PATH',
                        help='Path for trained model checkpoint to load')
    parser.add_argument('--seed', default=42, type=int, metavar='N',
                        help='Randomization seed number  (default: 42)')
    parser.add_argument('--prune', action='store_true',
                        help='Prune given model')
    parser.add_argument('--pruning_ratio', default=0, type=float, metavar='N',
                        help='Pruning Ratio to prune model')
    parser.add_argument('--pruning_type', default='global', type=str,
                        help='Pruning type: "global", "local"')
    parser.add_argument('--model_name', default=None, type=str,
                        help='Save Model filename location')
    parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument("--optimizer", type=str, default="sgd", 
                        choices=("sgd", "adam", "rmsprop"))
    parser.add_argument("--warmup-epochs", type=int, default=0, 
                        help="Number of warmup epochs")
    parser.add_argument("--num-classes", type=int, default=1000, 
                        help="Number of output classes in the model")
    parser.add_argument("--resume", type=str, default="",
                        help="path to latest checkpoint (default:None)")
    parser.add_argument( "--lr-schedule", type=str, default="cosine", choices=("step", "cosine"),
                        help="Learning rate schedule")

    # PixMix arguments
    parser.add_argument('--aug-severity', default=1, type=int,
                        help='Severity of base augmentation operators')
    parser.add_argument( '--beta', default=4, type=int, 
                        help='Severity of mixing')
    parser.add_argument('--k_mixing', default=4,type=int,
                        help='Mixing iterations')
    parser.add_argument('--all-ops', '--all', action='store_true', default=True,
                        help='Turn on all augmentation operations (+brightness,contrast,color,sharpness).')
    # CutOut arguments
    parser.add_argument('--cutout_length', default=56, type=int, metavar='N',
                        help='length of hole (usually half of dim)')
    

    #Score Based Pruning
    parser.add_argument(
        "--layer-type", type=str, choices=("dense", "subnet"), help="dense | subnet"
    )
    parser.add_argument(
        "--scaled-score-init",
        action="store_true",
        default=False,
        help="Init importance scores proportaional to weights (default kaiming init)",
    )
    parser.add_argument(
        "--exp-mode",
        type=str,
        choices=("pretrain", "prune", "finetune"),
        default="prune",
        help="Train networks following one of these methods.",
    )
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        default=False,
        help="freeze batch-norm parameters in pruning",
    )
    parser.add_argument(
        "--scores_init_type",
        choices=("kaiming_normal", "kaiming_uniform", "xavier_uniform", "xavier_normal", "weight_magnitude"),
        help="Which init to use for relevance scores",
    )
    parser.add_argument(
        "--init_type",
        choices=("kaiming_normal", "kaiming_uniform", "signed_const"),
        help="Which init to use for weight parameters: kaiming_normal | kaiming_uniform | signed_const",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=1.0,
        help="Fraction of weight variables kept in subnet",
    )
    parser.add_argument(
        "--save-dense",
        action="store_true",
        default=False,
        help="Save dense model alongwith subnets.",
    )

    return parser.parse_args()
