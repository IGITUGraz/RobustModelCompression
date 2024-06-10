
import numpy as np
import torch
import torch.nn.utils.prune as prune


def create_mask_global_lwm(model, pruning_ratio):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append( (module, 'weight') )

    prune.global_unstructured(parameters_to_prune,  pruning_method = prune.L1Unstructured, amount=pruning_ratio) #0.8

    return model, parameters_to_prune

def create_mask_local_lwm(model, pruning_ratio):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio) 
            parameters_to_prune.append( (module, 'weight') )
            
        elif isinstance(module, torch.nn.Linear): 
            prune.l1_unstructured(module, name='weight', amount=pruning_ratio) 
            parameters_to_prune.append( (module, 'weight') )

    return model, parameters_to_prune
