import model.backbone.resnet as resnet

import torch
from torch import nn
import torch.nn.functional as F
from functools import partial


class SSCDModel(nn.Module):
    def __init__(self, cfg):
        super(SSCDModel, self).__init__()

        # resnet50
        self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True, replace_stride_with_dilation=cfg['replace_stride_with_dilation'])

        channels = [128, 256, 512, 1024]

        self.conv_diff1 = nn.Sequential(nn.Conv2d(channels[0] * 2, channels[0], 1, 1, 0), nn.BatchNorm2d(channels[0]), nn.ReLU(True),
                                   nn.Conv2d(channels[0], channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU(True))
        self.conv_diff2 = nn.Sequential(nn.Conv2d(channels[1] * 2, channels[1], 1, 1, 0), nn.BatchNorm2d(channels[1]), nn.ReLU(True),
                                   nn.Conv2d(channels[1], channels[1], 3, 1, 1), nn.BatchNorm2d(channels[1]), nn.ReLU(True))
        self.conv_diff3 = nn.Sequential(nn.Conv2d(channels[2] * 2, channels[2], 1, 1, 0), nn.BatchNorm2d(channels[2]), nn.ReLU(True),
                                   nn.Conv2d(channels[2], channels[2], 3, 1, 1), nn.BatchNorm2d(channels[2]), nn.ReLU(True))
        self.conv_diff4 = nn.Sequential(nn.Conv2d(channels[3] * 2, channels[3], 1, 1, 0), nn.BatchNorm2d(channels[3]), nn.ReLU(True),
                                   nn.Conv2d(channels[3], channels[3], 3, 1, 1), nn.BatchNorm2d(channels[3]), nn.ReLU(True))

        # neck
        self.head = ASPPModule(channels[3],[6,12,18])

        # block combination of CNN and transformer
        num_heads = [1, 1, 1, 1]
        mlp_ratios = [2, 2, 2, 2]
        self.block3 = CBFFBlock(channels[3],num_heads[3],mlp_ratios[3])
        self.block2 = CBFFBlock(channels[2],num_heads[2],mlp_ratios[2])
        self.block1 = CBFFBlock(channels[1],num_heads[1],mlp_ratios[1])

        self.upcat4 = Up_and_cat(channels[3]//2,channels[3])
        self.upcat3 = Up_and_cat(channels[3],channels[2])
        self.upcat2 = Up_and_cat(channels[2],channels[1])
        self.upcat1 = Up_and_cat(channels[1],channels[0])

        self.cls_Cnn = nn.Sequential(CNN_block(channels[0]),
                                     nn.Conv2d(channels[0], channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU(True),
                                     nn.Conv2d(channels[0], 2, 1, bias=True))

        self.cls_Trans = nn.Sequential(Transformer_block(channels[0], num_heads[0], mlp_ratios[0]), 
                                       nn.Conv2d(channels[0], channels[0], 3, 1, 1), nn.BatchNorm2d(channels[0]), nn.ReLU(True),
                                       nn.Conv2d(channels[0], 2, 1, bias=True))

    def forward(self, x1, x2):
        h, w = x1.shape[-2:]

        feats1 = self.backbone.base_forward(x1)
        [c11, c12, c13, c14] = feats1
        
        feats2 = self.backbone.base_forward(x2)
        [c21, c22, c23, c24] = feats2

        d1 = self.conv_diff1((c11 - c21).abs())
        d2 = self.conv_diff2((c12 - c22).abs())
        d3 = self.conv_diff3((c13 - c23).abs())
        d4 = self.conv_diff4((c14 - c24).abs())

        h4 = self.head(d4)
        d4 = self.upcat4(h4, d4)
        f4 = self.block3(d4) 
        d3 = self.upcat3(f4, d3)
        f3 = self.block2(d3) 
        d2 = self.upcat2(f3, d2)
        f2 = self.block1(d2)
        d1 = self.upcat1(f2, d1)
        
        cnn_out = self.cls_Cnn(d1)
        cnn_out = F.interpolate(cnn_out, size=(h, w), mode="bilinear", align_corners=True)

        trans_out = self.cls_Trans(d1)
        trans_out = F.interpolate(trans_out, size=(h, w), mode="bilinear", align_corners=True)

        return cnn_out, trans_out


class Up_and_cat(nn.Module):
    def __init__(self, channels_in, channels_out):
        super(Up_and_cat, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(channels_out + channels_in, channels_out, 1, 1, 0), nn.BatchNorm2d(channels_out), nn.ReLU(True),
                                        nn.Conv2d(channels_out, channels_out, 3, 1, 1), nn.BatchNorm2d(channels_out), nn.ReLU(True))

    def forward(self, c_down, c_up):  # channels = [128,256,512,1024]
        c_down = F.interpolate(c_down, size=c_up.shape[-2:], mode="bilinear", align_corners=True)
        c_up = torch.cat([c_down, c_up], dim=1)
        c_cnn = self.conv_block(c_up)
        return c_cnn


class CBFFBlock(nn.Module):
    def __init__(self, channels_out, num_heads, mlp_ratios):
        super(CBFFBlock, self).__init__()
        qkv_bias = False
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.

        self.conv_block = nn.Sequential(nn.Conv2d(channels_out, channels_out, 3, 1, 1), nn.BatchNorm2d(channels_out), nn.ReLU(True),
                                        nn.Conv2d(channels_out, channels_out, 3, 1, 1), nn.BatchNorm2d(channels_out), nn.ReLU(True))
        self.trans_block = TransBlock(dim=channels_out, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)
        self.conv_out = nn.Sequential(nn.Conv2d(channels_out, channels_out, 3, 1, 1), nn.BatchNorm2d(channels_out), nn.ReLU(True))


    def forward(self, f_in):  # channels = [128,256,512,1024]
        c_cnn = self.conv_block(f_in)
        B, C, H, W = f_in.shape
        t_trans = f_in.flatten(2).transpose(1, 2)
        t_trans = self.trans_block(t_trans).transpose(1, 2).view(B, C, H, W)
        f_out = c_cnn + t_trans
        f_out = self.conv_out(f_out)
        return f_out


class CNN_block(nn.Module):
    def __init__(self, channels_out):
        super(CNN_block, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(channels_out, channels_out, 3, 1, 1), nn.BatchNorm2d(channels_out), nn.ReLU(True),
                                        nn.Conv2d(channels_out, channels_out, 3, 1, 1), nn.BatchNorm2d(channels_out), nn.ReLU(True))

    def forward(self, c_in): 
        c_out = self.conv_block(c_in)
        return c_out


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):  
        # input: shape [b,h*w,c]
        # output: shape [b,h*w,c]
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):  
        # input: shape [b,h*w,c]
        # output: shape [b,h*w,c]
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0:
            keep_prob = 1.0 - self.drop_prob
            random_tensor = keep_prob + torch.rand(x.size(0), 1, 1, 1, device=x.device)
            random_tensor = random_tensor.floor()  # 计算丢弃的层
            output = x / keep_prob * random_tensor
        else:
            output = x
        return output


class TransBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        # input: shape [b,h*w,c]
        # output: shape [b,h*w,c]
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

        
class Transformer_block(nn.Module):
    def __init__(self, embed_dims_out, num_heads, mlp_ratios):
        super(Transformer_block, self).__init__()
        qkv_bias = False
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.
        self.trans_block = TransBlock(dim=embed_dims_out, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                                       qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

    def forward(self, t_in):  # input: shape [b,c,h,w]
        B, C, H, W = t_in.shape
        t_in = t_in.flatten(2).transpose(1, 2)
        t_trans = self.trans_block(t_in).transpose(1, 2).view(B, C, H, W)
        return t_trans


class Transformer_head(nn.Module):
    def __init__(self, embed_dims_in, num_heads, mlp_ratios):
        super(Transformer_head, self).__init__()
        # embed_dims = [128,256,512,1024]
        qkv_bias = False
        qk_scale = None
        drop_rate = 0.
        attn_drop_rate = 0.
        drop_path_rate = 0.

        self.trans_block = TransBlock(dim=embed_dims_in, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
                                      qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=drop_path_rate)

    def forward(self, t_up):
        # input: shape [b,c,h,w]
        B, C, H, W = t_up.shape
        t_up = t_up.flatten(2).transpose(1, 2)
        t_trans = self.trans_block(t_up).transpose(1, 2).view(B, C, H, W)

        return t_trans


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 2
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)



