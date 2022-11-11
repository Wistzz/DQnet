# -*- coding: utf-8 -*-
# @Time    : 2020/12/19
# @Author  : Lart Pang
# @GitHub  : https://github.com/lartpang
from tokenize import group
import types

from torch import nn
from torch.optim import Adam, AdamW, SGD
import utils.lr_decay as lrd
from utils.adaxw import AdaXW

def get_optimizer(mode, params, initial_lr, optim_cfg):
    if mode == "sgd":
        optimizer = SGD(params=params, lr=initial_lr, **optim_cfg)
    elif mode == "adamw":
        optimizer = AdamW(params=params, lr=initial_lr, **optim_cfg)
    elif mode == "adam":
        optimizer = Adam(params=params, lr=initial_lr, **optim_cfg)
    elif mode == "adaxw":
        optimizer = AdaXW(params=params, lr=initial_lr, **optim_cfg)
    else:
        raise NotImplementedError
    return optimizer


def group_params(model, group_mode, initial_lr, optim_cfg):
    if group_mode == "yolov5":
        """
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # biases
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # no decay
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # apply decay

        if opt.adam:
            optimizer = optim.Adam(norm, lr=hyp["lr0"], betas=(hyp["momentum"], 0.999))  # adjust beta1 to momentum
        else:
            optimizer = optim.SGD(norm, lr=hyp["lr0"], momentum=hyp["momentum"], nesterov=True)

        optimizer.add_param_group({"params": weight, "weight_decay": hyp["weight_decay"]})  # add weight with weight_decay
        optimizer.add_param_group({"params": bias})  # add bias (biases)
        """
        norm, weight, bias = [], [], []  # optimizer parameter groups
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                bias.append(v.bias)  # conv bias and bn bias
            if isinstance(v, nn.BatchNorm2d):
                norm.append(v.weight)  # bn weight
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                weight.append(v.weight)  # conv weight
        params = [
            {"params": bias, "weight_decay": 0.0},
            {"params": norm, "weight_decay": 0.0},
            {"params": weight},
        ]
    elif group_mode == "r3":
        params = [
            # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
            # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
            # 到减少模型过拟合的效果。
            {
                "params": [param for name, param in model.named_parameters() if name[-4:] == "bias"],
                "lr": 2 * initial_lr,
                "weight_decay": 0,
            },
            {
                "params": [param for name, param in model.named_parameters() if name[-4:] != "bias"],
                "lr": initial_lr,
                "weight_decay": optim_cfg["weight_decay"],
            },
        ]
    elif group_mode == "all":
        params = model.parameters()
    elif group_mode == "finetune":
        if hasattr(model, "module"):
            model = model.module
        assert hasattr(model, "get_grouped_params"), "Cannot get the method get_grouped_params of the model."
        params_groups = model.get_grouped_params()
        params = [
            {"params": params_groups["space_encoder"], "lr": 5e-4},#/10/4},
            # {"params": params_groups["vit_encoder"], "lr": 5e-2},
            {"params": params_groups["other"], "lr": 5e-4},#/10/4},
        ]
    else:
        raise NotImplementedError
    return params


def construct_optimizer(model, initial_lr, mode, group_mode, cfg):
    params_groups_decoder = group_params(model, group_mode=group_mode, initial_lr=initial_lr, optim_cfg=cfg)

    if hasattr(model, "module"):
        model_without_ddp = model.module
    else:
        model_without_ddp = model
    params_groups_vit = lrd.param_groups_lrd(model_without_ddp.higher_encoder, 0.05,
        no_weight_decay_list=model_without_ddp.higher_encoder.no_weight_decay(),
        layer_decay=0.75
    )
    for  param_group in params_groups_vit:
        if 'lr_scale' in param_group:
            param_group['lr'] = initial_lr * param_group['lr_scale']
        else:
            param_group['lr'] = initial_lr

    params = params_groups_decoder + params_groups_vit

    optimizer = get_optimizer(mode=mode, params=params, initial_lr=initial_lr, optim_cfg=cfg)
    optimizer.lr_groups = types.MethodType(get_lr_groups, optimizer)
    optimizer.lr_string = types.MethodType(get_lr_strings, optimizer)

    return optimizer


def get_lr_groups(self):
    return [group["lr"] for group in self.param_groups]


def get_lr_strings(self):
    return ",".join([f"{group['lr']:10.3e}" for group in self.param_groups])
