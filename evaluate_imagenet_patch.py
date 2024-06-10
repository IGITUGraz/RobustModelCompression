import argparse
import gzip
import pickle
import os
import random
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils.apply_patch import ApplyPatch
from utils.loaders import CustomImageFolder
from utils.metrics_sparsity import output_sparsity


parser = argparse.ArgumentParser(description='PyTorch Model Training Codebase')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--test_all', action='store_true',
                    help='Run all validation set (default: will run 5k test samples from RobustBench)')
parser.add_argument('--dir', default='data/imagenet/validation/', type=str, metavar='DIR',
                    help='Path to dataset')
parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                    help='Evaluation mini-batch size (default: 1)')
parser.add_argument('--path', default=None, type=str, metavar='PATH',
                    help='Path for trained model checkpoint to load')
parser.add_argument('--save_path', default=None, type=str,
                    help='Patched image samples saving path')
parser.add_argument('--patch', default='adv', type=str,
                    help='Patch type: "adv", "rand", "black", "white", "bernoulli", "mean", "uniform", "srs_rn18"')
parser.add_argument('--translate_more', action='store_true',
                    help='play with the translation range of applied patch')
parser.add_argument('--scale_more', action='store_true',
                    help='play with the scale range of applied patch')
parser.add_argument('--verbose', action='store_true',
                    help='Print text stuff all the time')
parser.add_argument('--seed', default=42, type=int, metavar='N',
                    help='Randomization seed number  (default: 42)')

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parser.parse_args()
    set_all_seed(args.seed)

    # dictionary with the ImageNet label names
    with open(os.path.join(os.getcwd(), "assets/imagenet1000_clsidx_to_labels.txt")) as f:
        target_to_classname = eval(f.read())

    # Load adversarial patches
    with gzip.open(os.path.join(os.getcwd(), "assets/imagenet_patch.gz"), 'rb') as f:
        imagenet_patch = pickle.load(f)
    patches, targets, info = imagenet_patch
    num_patches = patches.shape[0]
    c, h, w = info['input_shape']

    for i in range(num_patches):
        if args.patch == "adv":
            pass
        elif args.patch == "srs_rn18":
            sparse_rs_patches = torch.load(os.path.join(os.getcwd(), "assets/resnet18_best_sparse_rs_patches.tar"))
            patches[i, ...] = sparse_rs_patches[i]
        elif args.patch == "uniform":
            patches[i, ...] = torch.as_tensor(np.random.uniform(size=(c, h, w)))
        elif args.patch == "rand":
            patches[i, ...] = torch.as_tensor(np.random.randn(c, h, w))
        elif args.patch == "black":
            patches[i, ...] = 0     # in practice, results in a test set of 5k not 50k
        elif args.patch == "white":
            patches[i, ...] = 1     # in practice, results in a test set of 5k not 50k
        elif args.patch == "bernoulli":
            patches[i, ...] = torch.cat(c * [torch.as_tensor(np.random.binomial(n=1, p=.5, size=(1, h, w)))])
        elif args.patch == "mean":
            mean_r, mean_g, mean_b = torch.mean(patches[i][:, 87:137, 87:137], dim=[1, 2])
            patches[i, 0, ...], patches[i, 1, ...], patches[i, 2, ...] = mean_r, mean_g, mean_b
        else:
            raise NotImplementedError

    # Choose an integer in the range 0-9 to select the patch
    correct_clean = 0
    correct_adv = 0
    n_success = 0
    n_samples = 0
    n_samples_patched = 0

    # Load the dataset
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])
    dataset = ImageFolder(args.dir, preprocess) if args.test_all else CustomImageFolder(args.dir, preprocess)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    print(len(dataset))

    # Load model
    model = torchvision.models.__dict__[args.arch](pretrained=args.pretrained)
    if args.path is not None:
        if 'salman' in args.path:
            checkpoint = torch.load(args.path)['model']
            new_model_state = {}
            for key in checkpoint.keys():
                if 'module.model.model' in key:
                    new_model_state[key[19:]] = checkpoint[key]
                else:
                    if 'module.model' in key:
                        new_model_state[key[13:]] = checkpoint[key]
            model.load_state_dict(new_model_state)
        else:
            checkpoint = torch.load(args.path)['state_dict']
            try:
                model.load_state_dict(checkpoint)
            except:
                new_model_state = {}
                for key in checkpoint.keys():
                    if key[:7] == 'module.':
                        new_model_state[key[7:]] = checkpoint[key]
                    else:
                        new_model_state[key[9:]] = checkpoint[key]
                model.load_state_dict(new_model_state)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()

    print("Sparsity of loaded model:")
    output_sparsity(model)
   
    # Apply Patches and evaluate
    plot_samples = torch.empty((num_patches, 5, c, h, w))
    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            x_clean = normalizer(x)
            pred_clean = model(x_clean)
            correct_clean += (pred_clean.argmax(1) == y).sum().item()
            n_samples += x.shape[0]

            for patch_id in range(num_patches):
                patch = patches[patch_id].clone()
                target = targets[patch_id].clone()

                if torch.cuda.is_available():
                    patch, target = patch.cuda(), target.cuda()

                if patch_id == 0:
                    # original
                    apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                                             translation_range=(.2, .2),
                                             rotation_range=(-45, 45),
                                             scale_range=(0.7, 1))
                    if args.scale_more:
                        apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                                                 translation_range=(.4, .4),
                                                 rotation_range=(-60, 60),
                                                 scale_range=(0.5, 1))
                    if args.translate_more:
                        apply_patch = ApplyPatch(patch, patch_size=info['patch_size'],
                                                 translation_range=(.4, .4),
                                                 rotation_range=(-60, 60),
                                                 scale_range=(0.7, 1))
                else:
                    apply_patch.set_patch(patch)

                patch_normalizer = Compose([apply_patch, normalizer])
                x_adv = patch_normalizer(x)
                pred_adv = model(x_adv)
                correct_adv += (pred_adv.argmax(1) == y).sum().item()
                n_success += (pred_adv.argmax(1) == target).sum().item()
                n_samples_patched += x.shape[0]


    clean_accuracy = 100*(correct_clean / n_samples)
    robust_accuracy = 100*(correct_adv / n_samples_patched)
    success_rate = 100*(n_success / n_samples_patched)

    print("Evaluated the model: {}".format(args.arch))
    print("Location of the model: {}".format(args.path))
    print("Was it pretrained?: {}".format(args.pretrained))
    print("Embedded attacker patch type: {}".format(args.patch))
    print("Total number of tested samples: {}".format(n_samples))
    print("Total number of tested samples with patches: {}".format(n_samples_patched))
    print("Clean accuracy: {:>0.2f}".format(clean_accuracy))
    print("Robust accuracy: {:>0.2f}".format(robust_accuracy))
    print("Target success rate: {:>0.2f}".format(success_rate))

if __name__ == '__main__':
    main()
