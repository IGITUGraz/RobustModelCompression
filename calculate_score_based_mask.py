import os
import numpy as np
import random
import copy
import models
from utils.train import train as trainer
from utils.eval import base as val
import data.loader as data_loader_aug
import torch
import torch.nn as nn
from args import parse_args
from utils.schedules import get_lr_policy, get_optimizer
from utils.logging import save_checkpoint
from score_based.model import get_layers, prepare_model, initialize_scaled_score, current_model_pruned_fraction, sanity_check_paramter_updates

# Parts used from  https://github.com/inspire-group/hydra

def set_all_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def main():
    args = parse_args()
    set_all_seed(args.seed)
    master_path = args.dir

    #Create model
    cl, ll = get_layers(args.layer_type)
    
    model = models.__dict__[args.arch](
            cl, ll, args.init_type, num_classes=args.num_classes
    )

    prepare_model(model, args)

    # Dataloader
    D = data_loader_aug.DataLoaderAugmentation(args, master_path)
    train_loader, val_loader = D.get_data_loaders()

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, args)
    lr_policy = get_lr_policy(args.lr_schedule)(optimizer, args)

    # Load source_net (if checkpoint provided). Only load the state_dict (required for pruning and fine-tuning)
    if args.path:
        if os.path.isfile(args.path):
            checkpoint = torch.load(args.path)
            model.load_state_dict(
                checkpoint["state_dict"], strict=False
            )  # allows loading dense models
        else:
            print("no checkpoint found")


    if args.scaled_score_init:
        initialize_scaled_score(model)

    assert not (args.path and args.resume), (
        "Incorrect setup: "
        "resume => required to resume a previous experiment (loads all parameters)|| "
        "source_net => required to start pruning/fine-tuning from a source model (only load state_dict)"
    )
    # resume (if checkpoint provided). Continue training with preiovus settings.
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_prec1 = checkpoint["best_prec1"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("no checkpoint found")

    best_prec1 = 0

    # Do not select source-net as last checkpoint as it might even be a dense model.
    # Most other function won't works well with a dense layer checkpoint.
    last_ckpt = copy.deepcopy(model.state_dict())

    # Start training
    for epoch in range(args.start_epoch, args.epochs + args.warmup_epochs):
        # adjust learning rate
        lr_policy(epoch)  

        trainer(
            train_loader,
            model,    
            criterion,
            optimizer,
            epoch,
            args
        )

        prec1, _ = val(model, val_loader, criterion, args, epoch)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "best_prec1": best_prec1,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            args,
            result_dir=os.path.join(master_path, "checkpoint"),
            save_dense=args.save_dense,
        )

        # Check what parameters got updated in the current epoch.
        sw, ss = sanity_check_paramter_updates(model, last_ckpt)
        print(
            f"Sanity check (exp-mode: {args.exp_mode}): Weight update - {sw}, Scores update - {ss}"
        )

    current_model_pruned_fraction(
        model, os.path.join(master_path, "checkpoint"), args, verbose=True
    )


if __name__ == "__main__":
    main()
