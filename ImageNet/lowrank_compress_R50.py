
import numpy as np

import torch
from Tucker_layer import tucker_conv_layer

from torch import nn
from typing import Callable, Dict, List, NewType, Optional, Set, Union

RecursiveReplaceFn = NewType("RecursieReplaceFn", Callable[[torch.nn.Module, torch.nn.Module, int, str, str], bool])


def prefix_name_lambda(prefix: str) -> Callable[[str], str]:
    """Returns a function that preprends `prefix.` to its arguments.

    Parameters:
        prefix: The prefix that the return function will prepend to its inputs
    Returns:
        A function that takes as input a string and prepends `prefix.` to it
    """
    return lambda name: (prefix + "." + name) if prefix else name


def replace_child(model:torch.nn.Module, child_name: str, compressed_child_model: torch.nn.Module, idx:int):
    if isinstance(model,torch.nn.Sequential):
        model[idx] = compressed_child_model
    elif isinstance(model, torch.nn.ModuleDict):
        model[child_name] = compressed_child_model
    else:
        model.add_module(child_name, compressed_child_model)


def apply_recursively_to_model(fn: RecursiveReplaceFn, model: torch.nn.Module, prefix: str = ""):

    get_prefixed_name = prefix_name_lambda(prefix)
    for idx, named_child in enumerate(model.named_children()):
        child_name, child = named_child
        child_prefixed_name = get_prefixed_name(child_name)
        if fn(model,child,idx,child_name, child_prefixed_name):
            continue
        else:
            apply_recursively_to_model(fn,child,child_prefixed_name)


def lowrank_compress_model(model: nn.Module, sparse_ratio=0.01):

    def _compress_and_replace_layer(parent: nn.Module, child: nn.Module, idx: int, name: str, prefixed_child_name: str):
        assert isinstance(parent, torch.nn.Module)
        assert isinstance(child, torch.nn.Module)
        print(name)
        if name == "conv1st" or name == "downsample" or name == "conv1" or name == "conv3":
             return True
        # if name == "downsample":
        #     return True
        # elif name == "0" and isinstance(child,torch.nn.Conv2d):
        #     return True
        elif isinstance(child,torch.nn.Conv2d) :
            compressed_child = tucker_conv_layer(child, sparse_ratio)
            replace_child(parent,name,compressed_child,idx)
            return True
        elif isinstance(child,torch.nn.Linear):
            return True
        else:
            return False

    apply_recursively_to_model(_compress_and_replace_layer, model)
    return model


# tensor = torch.ones(10,20,3,3)
# core,factors = conv_weight_mat_tucker_decomposition(tensor)
# print(core.shape)
# print(factors[1])

