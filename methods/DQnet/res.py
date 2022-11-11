import timm
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
from timm.models.resnet import Bottleneck
from timm.models.vision_transformer import Mlp
import numpy as np

class ResNet(timm.models.resnet.ResNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)  
        self.cross0 = WindowFusion(768) 
        self.cross1 = WindowFusion(768)
        self.cross2 = WindowFusion(768)
        self.cross3 = WindowFusion(768)  

        self.convert0 = nn.Sequential(
            nn.ConvTranspose2d(768, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        )
        self.convert1 = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
        )
        self.convert2 = nn.ConvTranspose2d(768, 512, kernel_size=2, stride=2)
        self.convert3 = nn.Conv2d(768, 1024, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.down0 = nn.Conv2d(768, 64, kernel_size=1, stride=1, padding=0)
        self.down1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.down2 = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0)
        self.down3 = nn.Conv2d(768, 1024, kernel_size=1, stride=1, padding=0)        
        self.proj0 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4), nn.Conv2d(64, 768, kernel_size=1, stride=1, padding=0))
        self.proj1 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4), nn.Conv2d(256, 768, kernel_size=1, stride=1, padding=0))
        self.proj2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(512, 768, kernel_size=1, stride=1, padding=0))
        self.proj3 = nn.Sequential(nn.Conv2d(1024, 768, kernel_size=1, stride=1, padding=0))     
        self.norm0 = nn.LayerNorm(768)
        self.norm1 = nn.LayerNorm(768)
        self.norm2 = nn.LayerNorm(768)
        self.norm3 = nn.LayerNorm(768)
        self.pos_norm0 = nn.BatchNorm2d(64)
        self.pos_norm1 = nn.BatchNorm2d(256)
        self.pos_norm2 = nn.BatchNorm2d(512)
        self.pos_norm3 = nn.BatchNorm2d(1024)

    def pos(self, x, pos_token):
        B, HW, C = x.shape
        H = int(math.sqrt(HW))
        pos_token = F.interpolate(
            pos_token,
            size=(H // 24, H // 24),
            mode='bicubic', align_corners=False
        )
        pos_token = pos_token.permute(0, 2, 3, 1).flatten(1, 2)
        return x + pos_token

    def forward_features(self, x, y):
        features = []
        atts = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        features.append(x)
        x = self.maxpool(x) #c=64


        # # patch embed -> cross attention
        ys = [yy.flatten(2).transpose(1,2) for yy in y]
        xx = self.proj0(x)
        xx = xx.flatten(2).transpose(1, 2)
        xx = self.norm0(xx) # 20, 576, 768
        b,hh,c = xx.shape
        h = int(math.sqrt(hh))
        cr, att = self.cross0(ys[3], xx)
        atts.append(att)
        feat0 = self.relu(self.pos_norm0(self.convert0(cr)))
        x = self.layer1(feat0)
        features.append(x)

        xx = self.proj1(x)
        xx = xx.flatten(2).transpose(1, 2)
        xx = self.norm1(xx)
        cr, att = self.cross1(ys[3], xx)
        atts.append(att)
        feat1 = self.relu(self.pos_norm1(self.convert1(cr)))
        x = self.layer2(feat1)
        features.append(x)

        xx = self.proj2(x)
        xx = xx.flatten(2).transpose(1, 2)
        xx = self.norm2(xx)      
        cr, att = self.cross2(ys[3], xx)
        atts.append(att)
        feat2 = self.relu(self.pos_norm2(self.convert2(cr)))
        x = self.layer3(feat2)
        features.append(x)

        xx = self.proj3(x)
        xx = xx.flatten(2).transpose(1, 2)
        xx = self.norm3(xx)      
        cr, att = self.cross3(ys[3], xx)
        atts.append(att)
        feat3 = self.relu(self.pos_norm3(self.convert3(cr)))
        x = self.layer4(feat3)
        features.append(x)


        return features, atts, [feat0, feat1, feat2, feat3]

    def plain_forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        features.append(x)
        x = self.maxpool(x) #c=64

        # vit features blend in
        x = self.layer1(x) # 1/4
        features.append(x)
        x = self.layer2(x) # 1/4
        features.append(x)
        x = self.layer3(x) # 1/8
        features.append(x)
        x = self.layer4(x) # 1/16
        features.append(x)

        return features
        
    def forward(self, x, y=None, feedback=False):
        if feedback:
            features = self.forward_features(x, y)
        else:
            features = self.plain_forward(x)
        return features

def resnet50(**kwargs):
    model = ResNet(
        block=Bottleneck, layers=[3, 4, 6, 3],  
        **kwargs
    )
    pretrain = './resnet50_pretrain.pth'
    checkpoint = torch.load(pretrain, map_location='cpu')


    print("Load pre-trained checkpoint from: %s" % pretrain)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']


    # load pre-trained model
    msg = model.load_state_dict(checkpoint, strict=False)
    print(msg)
    return model



class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class CrossFusion(nn.Module):
    def __init__(self, dim, window_size=(18, 18), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # q_size = window_size[0]
        # kv_size = window_size[1]
        # rel_sp_dim = 2 * q_size - 1
        # self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        # self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)    
        # self.att_project = nn.Linear(num_heads, 1)    
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        # for att loss

        # self.bias_deconv = nn.Sequential(
        #     nn.ConvTranspose2d(768, 64, kernel_size=4, stride=4),
        #     Norm2d(64),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(64, 1, kernel_size=4, stride=4))

    def forward(self, x, y):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """


        B, N, C = x.shape
        H= int(math.sqrt(N))
        W= H
        x1 = x.reshape(B, H, W, C)
        identity = x1.permute(0,3,1,2)
        y1 = y.reshape(B, H, W, C)
        identity_y = y1.permute(0,3,1,2)
        x = x

        # pad_l = pad_t = 0
        # pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        # pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        # x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        # _, Hp, Wp, _ = x.shape

        # x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        # y = window_partition(y, self.window_size[0])
        # x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C
        # y = y.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C

        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 
        q = q[0] #nW*B, heads, window_size*window_size, C/heads(144,8,64,96)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))


        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)



        x = x.view(B, H * W, C)
        x = x.permute(0,2,1)
        x = x.view(B,C,H,W)
        # bias = self.bias_deconv(torch.abs(x)).sigmoid()
        # print(bias.shape, attn.shape)
        # return x+identity+identity, bias

class WindowFusion(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size=(10, 10), num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., attn_head_dim=None):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        q_size = window_size[0]
        kv_size = window_size[1]
        rel_sp_dim = 2 * q_size - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(rel_sp_dim, head_dim))

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)    
        # self.att_project = nn.Linear(num_heads, 1)    
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x,y):
        """ Forward function.
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """

        B_, N, C = x.shape
        H= int(math.sqrt(N))
        W= H
        x = x.reshape(B_, H, W, C)
        identity = x.permute(0,3,1,2)
        y = y.reshape(B_, H, W, C)
        identity_y = y.permute(0,3,1,2)
        x = x

        pad_l = pad_t = 0
        pad_r = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
        pad_b = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]

        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        y = F.pad(y, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x = window_partition(x, self.window_size[0])  # nW*B, window_size, window_size, C
        y = window_partition(y, self.window_size[0])
        x = x.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C
        y = y.view(-1, self.window_size[1] * self.window_size[0], C)  # nW*B, window_size*window_size, C
        B_w = x.shape[0]
        N_w = x.shape[1]
        kv = self.kv(y).reshape(B_w, N_w, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        q = self.q(x).reshape(B_w, N_w, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 
        q = q[0] #nW*B, heads, window_size*window_size, C/heads(144,8,64,96)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = calc_rel_pos_spatial(attn, q, self.window_size, self.window_size, self.rel_pos_h, self.rel_pos_w)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_w, N_w, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        x = x.view(-1, self.window_size[1], self.window_size[0], C)
        x = window_reverse(x, self.window_size[0], Hp, Wp)  # B H' W' C


        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B_, H * W, C)
        x = x.permute(0,2,1)
        x = x.view(B_,C,H,W)



        return x*identity+identity_y, x.sigmoid()#bias



def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def calc_rel_pos_spatial(
    attn,
    q,
    q_shape,
    k_shape,
    rel_pos_h,
    rel_pos_w,
    ):
    """
    Spatial Relative Positional Embeddings.
    """
    sp_idx = 0
    q_h, q_w = q_shape
    k_h, k_w = k_shape

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = (
        torch.arange(q_h)[:, None] * q_h_ratio - torch.arange(k_h)[None, :] * k_h_ratio
    )
    dist_h += (k_h - 1) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = (
        torch.arange(q_w)[:, None] * q_w_ratio - torch.arange(k_w)[None, :] * k_w_ratio
    )
    dist_w += (k_w - 1) * k_w_ratio

    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]

    B, n_head, q_N, dim = q.shape

    r_q = q[:, :, sp_idx:].reshape(B, n_head, q_h, q_w, dim)
    rel_h = torch.einsum("byhwc,hkc->byhwk", r_q, Rh)
    rel_w = torch.einsum("byhwc,wkc->byhwk", r_q, Rw)

    attn[:, :, sp_idx:, sp_idx:] = (
        attn[:, :, sp_idx:, sp_idx:].view(B, -1, q_h, q_w, k_h, k_w)
        + rel_h[:, :, :, :, :, None]
        + rel_w[:, :, :, :, None, :]
    ).view(B, -1, q_h * q_w, k_h * k_w)

    return attn


