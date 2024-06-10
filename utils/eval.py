import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
import torch.nn.functional as F
from utils.logging import AverageMeter, ProgressMeter
from scipy.stats import norm
import numpy as np
import time


CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
               'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
               'jpeg_compression']


def get_output_for_batch(model, img, temp=1):
    """
        model(x) is expected to return logits (instead of softmax probas)
    """
    with torch.no_grad():
        out = nn.Softmax(dim=-1)(model(img) / temp)
        p, index = torch.max(out, dim=-1)
    return p.data.cpu().numpy(), index.data.cpu().numpy()


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def base(model, val_loader, criterion, args, epoch=0):
    """
        Evaluating on unmodified validation set inputs.
    """

    print("we are here in evaluating base")
    
    batch_time = AverageMeter("Time", ":6.3f")
    losses = AverageMeter("Loss", ":.4f")
    top1 = AverageMeter("Acc_1", ":6.2f")
    top5 = AverageMeter("Acc_5", ":6.2f")
    progress = ProgressMeter(
        len(val_loader), [batch_time, losses, top1, top5], prefix="Test: "
    )

    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            images, target = data[0].cuda(), data[1].cuda()

            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % args.print_freq == 0:
                progress.display(i)

            if writer:
                progress.write_to_tensorboard(
                    writer, "test", epoch * len(val_loader) + i
                )

            # write a sample of test images to tensorboard (helpful for debugging)
            if i == 0 and writer:
                writer.add_image(
                    "test-images",
                    torchvision.utils.make_grid(images[0 : len(images) // 4]),
                )
        progress.display(i)  

    return top1.avg, top5.avg


def compute_mce(corruption_accs):
    alexnet_err = [88.6, 89.4, 92.3, 82.0, 82.6, 78.6, 79.8, 86.7, 82.7, 81.9, 56.5, 85.3, 64.6, 71.8, 60.7]
    mce = 0.
    for i in range(len(CORRUPTIONS)):
        avg_err = 100 - np.mean(corruption_accs[CORRUPTIONS[i]])
        ce = 100 * avg_err / alexnet_err[i]
        print(CORRUPTIONS[i], ce)
        mce += ce / 15
    return mce