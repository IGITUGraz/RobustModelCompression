import argparse
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.nn.functional as F
from args import parse_args
import torch.nn.utils.prune as prune
from utils.loaders import CustomImageFolder
from utils.logging import AverageMeter, ProgressMeter, save_checkpoint, save_model
from utils.eval import accuracy
from utils.train import train as trainer
import data.loader as data_loader_aug
from utils.metrics_sparsity import output_sparsity
from utils.calibration_tools import *
from magnitude_based.prune import create_mask_global_lwm, create_mask_local_lwm


def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    to_np = lambda x: x.data.to('cpu').numpy()

    confidence = []
    correct = []

    num_correct = 0

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda()
            target = target.cuda()

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            num_correct += pred.eq(target.data).sum().item()

            confidence.extend(to_np(F.softmax(output, dim=1).max(1)[0]).squeeze().tolist())
            pred = output.data.max(1)[1]
            correct.extend(pred.eq(target).to('cpu').numpy().squeeze().tolist())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
    
    print('RMS {:.3f}\n'.format(100 * calib_err(np.array(confidence.copy()), np.array(correct.copy()), p='2')))

    return losses.avg, top1.avg, top5.avg


def train_model (model, criterion, optimizer, args, master_path, parameters_to_prune):
    model.train()
    best_acc1 = 0

    # Get augmented input
    D = data_loader_aug.DataLoaderAugmentation(args, master_path)
    train_loader, val_loader = D.get_data_loaders()

    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (
                1 + np.cos(step / total_steps * np.pi))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,
            1e-6 / (args.lr * args.batch_size / 256.)))

    if args.start_epoch != 0:
        scheduler.step(args.start_epoch * len(train_loader))

    model_dir = "results/checkpoints_pruned/" + args.augmentation + "/"
    
    for epoch in range(args.start_epoch, args.epochs):
        print('Starting epoch %d / %d' % (epoch + 1, args.epochs))
        
        # train for one epoch
        train_losses_avg, train_top1_avg, train_top5_avg = trainer(train_loader, model, criterion, optimizer, epoch, args)
        scheduler.step()
        
        
        print("Evaluating on validation set")
        val_losses_avg, val_top1_avg, val_top5_avg = validate(val_loader, model, criterion, args)

        logname = args.model_name + "_training_log.csv"

        # Save results in log file
        with open(os.path.join(model_dir, logname), 'a') as f:
            f.write('%03d,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f,%0.5f\n' % (
                (epoch + 1),
                train_losses_avg, train_top1_avg, train_top5_avg,
                val_losses_avg, val_top1_avg, val_top5_avg
            ))

        # remember best acc@1 and save checkpoint
        is_best = val_top1_avg > best_acc1
        best_acc1 = max(val_top1_avg, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'parameters_to_prune:' : parameters_to_prune
        }, is_best, args, model_dir, args.model_name)



def test(model, args, master_path):
    model.eval()

    normalizer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = Compose([Resize(256), CenterCrop(224), ToTensor()])

    correct_clean = 0
    n_samples = 0
    master_path += 'imagenet/validation/'
    dataset = ImageFolder(master_path, preprocess) if args.test_all else CustomImageFolder(master_path, preprocess)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    with torch.no_grad():
        for i, (x, y) in enumerate(data_loader):
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()
            x_clean = normalizer(x)
            pred_clean = model(x_clean)
            correct_clean += (pred_clean.argmax(1) == y).sum().item()
            n_samples += x.shape[0]
    accuracy = 100 * (correct_clean / n_samples)

    print(f"Total number of tested samples: {n_samples}")
    print(f"Clean accuracy: {accuracy:>0.2f}%")


def main():
    args = parse_args()
    master_path = args.dir
    set_all_seed(args.seed)
    print("Pruning_Ratio: "  + str(args.pruning_ratio))

    # Load model
    model = torchvision.models.__dict__[args.arch](pretrained=args.pretrained)

    if args.path is not None:
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

    # Measure sparsity before pruning
    output_sparsity(model)

    # Test model before pruning
    test(model, args, master_path)

    # Apply pruning
    parameters_to_prune = 0
    if args.prune:        
        if args.pruning_type == "global":
            model, parameters_to_prune = create_mask_global_lwm(model, args.pruning_ratio)
        elif args.pruning_type == "local":
            model, parameters_to_prune = create_mask_local_lwm(model, args.pruning_ratio)
        else:
            raise NotImplementedError

        print("Sparsity AFTER pruning model:")
        output_sparsity(model)

    # Test without retraining model
    test(model, args, master_path)

    # Retrain remaining model parameters
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay, nesterov=True)
    
    train_model(model, criterion, optimizer, args, master_path, parameters_to_prune)

    # Prune model parameters based on mask
    if args.prune:  
        for param in parameters_to_prune:
            prune.remove(param[0], param[1])
     
    output_sparsity(model)

    # Test Pruned Model
    test(model, args, master_path)

    # Measure sparsity after pruning and retraining
    output_sparsity(model)

    model_dir = "results/checkpoints_pruned/" + args.augmentation + "/"
    save_model(model, model_dir, args.model_name)
    
    return


if __name__ == '__main__':
    main()
