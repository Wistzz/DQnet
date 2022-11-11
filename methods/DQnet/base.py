import numpy as np
import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from methods.module.base_model import BasicModelClass
from methods.module.conv_block import ConvBNReLU
from utils.builder import MODELS
from utils.ops import cus_sample
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from functools import partial
import math




def get_coef(iter_percentage, method):
    if method == "linear":
        milestones = (0.3, 0.7)
        coef_range = (0, 1)
        min_point, max_point = min(milestones), max(milestones)
        min_coef, max_coef = min(coef_range), max(coef_range)
        if iter_percentage < min_point:
            ual_coef = min_coef
        elif iter_percentage > max_point:
            ual_coef = max_coef
        else:
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
    elif method == "cos":
        coef_range = (0, 1)
        min_coef, max_coef = min(coef_range), max(coef_range)
        normalized_coef = (1 - np.cos(iter_percentage * np.pi)) / 2
        ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    else:
        ual_coef = 1.0
    return ual_coef


@MODELS.register()
class DQnet_BASE(BasicModelClass):
    def __init__(self):
        super().__init__()
        # for feature fusion
        self.last_ch = 128

    def encoder_translayer(self, x):
        higher_feature = self.higher_encoder(x)
        space_feature,_,_ = self.space_encoder(x, higher_feature, feedback=True)

        return space_feature, higher_feature#trans_feats

    def body(self, m_scale):
        space_feature, higher_feature = self.encoder_translayer(m_scale)
        over_all_feature = space_feature[1:]+higher_feature[3:]
        logits = self.segmentation_head(over_all_feature)
        return dict(seg=logits)

    def train_forward(self, data, **kwargs):

        output = self.body(
            m_scale=data["image1.0"],
        )
        loss, loss_str = self.cal_loss(
            all_preds=output,
            gts=data["mask"],
            iter_percentage=kwargs["curr"]["iter_percentage"],
        )
        return dict(sal=output["seg"].sigmoid()), loss, loss_str

    def test_forward(self, data, **kwargs):
        output = self.body(
            m_scale=data["image1.0"],
        )
        return output["seg"]


    def att_loss(self,pred,mask):
        batch_size = mask.shape[0]
        h = math.sqrt(pred.shape[-1])
        mask = F.interpolate(mask,size=(int(h),int(h)), mode='bilinear')
        x = mask.view(batch_size, 1, -1).permute(0, 2, 1) 
        g = x @ x.transpose(-2,-1) # b 28*28 28*28
        g = g.unsqueeze(1) # b 1 28*28 28*28
        attbce=F.binary_cross_entropy_with_logits(pred, g)
        return attbce

    def cal_loss(self, all_preds: dict, gts: torch.Tensor, method="cos", iter_percentage: float = 0):
        ual_coef = 0.1#get_coef(iter_percentage, method)

        losses = []
        loss_str = []
        # for main
        for name, preds in all_preds.items():
            resized_gts = cus_sample(gts, mode="size", factors=preds.shape[2:])
            
            # wbce and iou loss
            weit = 1 + 5*torch.abs(F.avg_pool2d(resized_gts, kernel_size=31, stride=1, padding=15) - resized_gts)
            wbce = F.binary_cross_entropy_with_logits(preds, resized_gts, reduce='none')
            wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

            preds = torch.sigmoid(preds)
            inter = ((preds * resized_gts)*weit).sum(dim=(2, 3))
            union = ((preds + resized_gts)*weit).sum(dim=(2, 3))
            wiou = 1 - (inter + 1)/(union - inter+1)
            overall_loss = (wbce + wiou).mean()

            losses.append(overall_loss)
            loss_str.append(f"{name}_structure_loss: {overall_loss.item():.5f}")

        return sum(losses), " ".join(loss_str)

    def get_grouped_params(self):
        param_groups = {}
        for name, param in self.named_parameters():
            if name.startswith("shared_encoder.layer"):
                param_groups.setdefault("pretrained", []).append(param)
            elif name.startswith("shared_encoder."):
                param_groups.setdefault("fixed", []).append(param)
            else:
                param_groups.setdefault("retrained", []).append(param)
        return param_groups







