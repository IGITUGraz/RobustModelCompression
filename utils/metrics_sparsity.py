import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import style

def count_zero_weights(model):
    zeros = 0
    non_zeros = 0
    for name, param in model.named_parameters():
        if param is not None:
            zeros += param.numel() - param.nonzero().size(0)
            non_zeros += param.nonzero().size(0)
    return zeros, non_zeros

def output_sparsity(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of Parameters: {}".format(pytorch_total_params))
    print("Number of trainable Parameters: {}".format(pytorch_total_trainable_params))
    print("Number of Parameters Set to Zero: {}".format(count_zero_weights(model)[0]))
    print("Number of Parameters Set: {}".format(count_zero_weights(model)[1]))
    measure_global_sparsity(model)

def measure_sparsity_per_layer(model):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            print(
            "Sparsity in " + name + ": {:.2f}%".format(100. * float(torch.sum(module.weight == 0))
            / float(module.weight.nelement())
            ))

def measure_global_sparsity(model):
    num_zero = 0
    num_elem = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            num_zero += (torch.sum(module.weight == 0))
            num_elem += module.weight.nelement()

    print("Global sparsity: {:.2f}%".format(100. * float(num_zero) / float(num_elem)))