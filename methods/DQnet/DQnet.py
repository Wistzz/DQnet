from .base import DQnet_BASE, ConvBNReLU
from .vit import vit_base_patch16_224
from utils.builder import MODELS
from torch import nn
import torch
import timm
from .decoder import SimpleHead, AuxHead
from .res import resnet50
import torch.nn.functional as F


@MODELS.register()
class DQnet(DQnet_BASE):
    def __init__(self):
        super().__init__()
        self.space_encoder = resnet50()
        self.higher_encoder = vit_base_patch16_224()
        self.segmentation_head = SimpleHead(
            num_classes=1
        )
        self.aux_head = AuxHead()
    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("space_encoder."):
                # pass
                param_groups.setdefault("space_encoder", []).append(param)
            elif name.startswith("higher_encoder"):
                param_groups.setdefault("vit_encoder", []).append(param)
            else:
                param_groups.setdefault("other", []).append(param)
        return param_groups
