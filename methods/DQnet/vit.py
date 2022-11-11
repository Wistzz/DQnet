import timm
import math
import torch
import torch.nn.functional as F
from .pos_embed import interpolate_pos_embed
import torch.nn as nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from .layers import Block
from functools import partial

class PatchEmbed(timm.models.layers.patch_embed.PatchEmbed):
    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



class ViT(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, depth=None, **kwargs):
        super().__init__(embed_layer=PatchEmbed, **kwargs)
        del self.head
        self.depth=depth

    def forward_features(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        pos_embed = self.pos_embed
        cls_tokens = pos_embed[:, :1, :]
        pos_tokens = pos_embed[:, 1:, :]
        pos_tokens = pos_tokens.reshape(1, *self.patch_embed.grid_size, -1).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(
            pos_tokens,
            size=(H // self.patch_embed.patch_size[0], W // self.patch_embed.patch_size[1]),
            mode='bicubic', align_corners=False
        )
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        pos_embed = torch.cat((cls_tokens, pos_tokens), dim=1)

        x = self.pos_drop(x + pos_embed)

        features=[]

        for i in range(self.depth):
            x = self.blocks[i](x)
            if i in [2,5,8,11]:           
                features.append(x)
        features[-1] = self.norm(features[-1])   
        return features


    def forward(self, x, strides=[4, 8, 16, 32]):

        features = self.forward_features(x)

        features = [x[:, 1:, :] for x in features]
        B, N, C = features[0].shape
        S = int(math.sqrt(N))
        features = [x.reshape(B, S, S, C).permute(0, 3, 1, 2) for x in features]

        return features



def vit_base_patch16_224(**kwargs):
    model = ViT(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path_rate=0.1,
        **kwargs
    )
    pretrain = './mae_pretrain_vit_base.pth'
    checkpoint = torch.load(pretrain, map_location='cpu')

    print("Load pre-trained checkpoint from: %s" % pretrain)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    state_dict = model.state_dict()

    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint)

    # load pre-trained model
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model



