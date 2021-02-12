from torch import nn


def freeze_part(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = False


def unfreeze_part(model: nn.Module):
    for par in model.parameters():
        par.requires_grad = True
