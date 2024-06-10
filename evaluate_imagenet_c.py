import argparse
import os
import random
import numpy
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from utils.loaders import CustomImageFolder
from utils.metrics_sparsity import output_sparsity

parser = argparse.ArgumentParser(description='PyTorch Model Training Codebase')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    help='model architecture (default: resnet50)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--test_all', action='store_true',
                    help='Run all validation set (default: will run 5k test samples from RobustBench)')
parser.add_argument('-c', '--corruption', action='store_true',
                    help='Corruptions to be used for test set evaluations')
parser.add_argument('--dir', default='data/', type=str, metavar='DIR',
                    help='Path to dataset')
parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                    help='Evaluation mini-batch size (default: 1)')
parser.add_argument('--path', default=None, type=str, metavar='PATH',
                    help='Path for trained model checkpoint to load')
parser.add_argument('--seed', default=42, type=int, metavar='N',
                    help='Randomization seed number  (default: 42)')

CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']

def compute_mce(corruption_accs):
    alexnet_err = [88.6, 89.4, 92.3, 82.0, 82.6, 78.6, 79.8, 86.7, 82.7, 81.9, 56.5, 85.3, 64.6, 71.8, 60.7]
    mce = 0.
    for i in range(len(CORRUPTIONS)):
        avg_err = 100 - numpy.mean(corruption_accs[CORRUPTIONS[i]])
        ce = 100 * avg_err / alexnet_err[i]
        print(CORRUPTIONS[i], ce)
        mce += ce / 15
    return mce

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    numpy.random.seed(seed)
    random.seed(seed)

def main():
    args = parser.parse_args()

    master_path = args.dir
    batch_size = args.batch_size
    set_all_seed(args.seed)
    pruning_ratio = args.pruning_ratio/10.0

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

    print("Evaluating the model: {}".format(args.arch))
    print("Location of the model: {}".format(args.path))
    print("Was it pretrained?: {}".format(args.pretrained))

    print("Sparsity of loaded model:")
    output_sparsity(model)

    #Evaluate Model
    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])

    # Load the dataset
    if args.corruption:
        master_path += 'imagenet-c'
        corruption_accs = {}
        for corr in CORRUPTIONS:
            print(corr)
            for sev in [1, 2, 3, 4, 5]: #over all severities
                correct_clean = 0
                n_samples = 0
                dataset = ImageFolder(os.path.join(master_path, corr, str(sev)), Compose([ToTensor()]))
                data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

                with torch.no_grad():
                    for i, (x, y) in enumerate(data_loader):
                        if torch.cuda.is_available():
                            x = x.cuda()
                            y = y.cuda()

                        x_clean = normalizer(x)                      
                        if args.no_normalization:
                            x_clean = x                  

                        pred_clean = model(x_clean)
                        correct_clean += (pred_clean.argmax(1) == y).sum().item()
                        n_samples += x.shape[0]
                accuracy = 100 * (correct_clean / n_samples)
                print(f"Severity: {sev}")
                print(f"Accuracy: {accuracy:>0.2f}%")
                if corr in corruption_accs:
                    corruption_accs[corr].append(accuracy)
                else:
                    corruption_accs[corr] = [accuracy]
        print(corruption_accs)

        corr_mean_accs = [numpy.mean(corruption_accs[CORRUPTIONS[i]]) for i in range(len(CORRUPTIONS))]
        print('Corrupted Set Accuracies: ')
        for i in range(len(CORRUPTIONS)):
            print(CORRUPTIONS[i], corr_mean_accs[i])
        mean_accs = sum(corr_mean_accs) / len(CORRUPTIONS)
        print('Mean: ', mean_accs)
        print('mCE (normalized by AlexNet): ', compute_mce(corruption_accs))

    else:
        correct_clean = 0
        n_samples = 0
        master_path += 'imagenet/validation/'
        dataset = ImageFolder(master_path, preprocess) if args.test_all else CustomImageFolder(master_path, preprocess)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

        with torch.no_grad():
            for i, (x, y) in enumerate(data_loader):
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
                
                    x_clean = normalizer(x)                      
                    if args.no_normalization:
                        x_clean = x    
                            
                pred_clean = model(x_clean)
                correct_clean += (pred_clean.argmax(1) == y).sum().item()
                n_samples += x.shape[0]
        accuracy = 100 * (correct_clean / n_samples)

        print(f"Total number of tested samples: {n_samples}")
        print(f"Clean accuracy: {accuracy:>0.2f}%")
            
    return


if __name__ == '__main__':
    main()
