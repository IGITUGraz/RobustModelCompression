import torch
import shutil
import os
from enum import Enum
from distutils.dir_util import copy_tree
from score_based.model import subnet_to_dense

def save_checkpoint(state, is_best, args, result_dir, filename="checkpoint.pth.tar", save_dense=False): 
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    
    torch.save(state, os.path.join(result_dir, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(result_dir, filename),
            os.path.join(result_dir, filename),
        )

    if save_dense:
        state["state_dict"] = subnet_to_dense(state["state_dict"], args.k)
        torch.save(
            subnet_to_dense(state, args.k),
            os.path.join(result_dir, "checkpoint_dense.pth.tar"),
        )

        if is_best:
            shutil.copyfile(
                os.path.join(result_dir, filename),
                os.path.join(result_dir, filename),
            )


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)


def save_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_filepath = os.path.join(model_dir, model_filename)
    torch.save({'state_dict': model.state_dict()}, model_filepath)


def create_subdirs(sub_dir):
    os.mkdir(sub_dir)
    os.mkdir(os.path.join(sub_dir, "checkpoint"))


def write_to_file(file, data, option):
    with open(file, option) as f:
        f.write(data)


def clone_results_to_latest_subdir(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)
    copy_tree(src, dst)


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

