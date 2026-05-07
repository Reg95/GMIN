'''
This repository is used to implement all upsamplers(only x4) and tools for Efficient SR
@author
    LI Zehyuan from SIAT
    LIU yingqi from SIAT
'''

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import basicsr.archs.Upsamplers as Upsamplers
from basicsr.utils.registry import ARCH_REGISTRY
from torch.nn import init
from torch import Tensor
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import numbers



####################################大核注意力机制#########################################################################
class LKAT(nn.Module):

    def __init__(self, n_feats):
        super().__init__()

        # self.norm = LayerNorm(n_feats, data_format='channels_first')
        # self.scale = nn.Parameter(torch.zeros((1, n_feats, 1, 1)), requires_grad=True)

        self.conv0 = nn.Sequential(nn.Conv2d(n_feats, n_feats, 1, 1, 0), nn.GELU())

        self.att = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, 7, 1, 7 // 2, groups=n_feats),
            nn.Conv2d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3, groups=n_feats, dilation=3),
            nn.Conv2d(n_feats, n_feats, 1, 1, 0))

        self.conv1 = nn.Conv2d(n_feats, n_feats, 1, 1, 0)

    def forward(self, x):
        x = self.conv0(x)
        x = x * self.att(x)
        x = self.conv1(x)
        return x

def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


##########################################################################
## Layer Norm


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFreeLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(BiasFreeLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBiasLayerNorm(nn.Module):

    def __init__(self, normalized_shape):
        super(WithBiasLayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):

    def __init__(self, dim, layer_norm_type):
        super(LayerNorm, self).__init__()
        if layer_norm_type == 'BiasFree':
            self.body = BiasFreeLayerNorm(dim)
        else:
            self.body = WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        # return to_4d(self.body(to_3d(x)), h, w)
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################

def build_act_layer(act_type):
    """Build activation layer."""
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

##########################################################################
class EFeedForward(nn.Module):
    def __init__(self, dim):
        super(EFeedForward, self).__init__()
        ffn_expansion_factor=2
        bias=False

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*3, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.pwconw = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x1, x2, x3 = self.project_in(x).chunk(3, dim=1)
        x2, x3 = self.dwconv(torch.cat((x2,x3),dim=1)).chunk(2, dim=1)
        x = F.gelu(x1)*x3 + F.gelu(x2)*x3 + self.pwconw(x3)
        x = self.project_out(x)
        return x

## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):

    def __init__(self, dim):
        super(FeedForward, self).__init__()
        ffn_expansion_factor=2
        bias=False
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=5,
            stride=1,
            padding=2,
            groups=hidden_features * 2,
            bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        # self.dwconv5 = nn.Sequential(
        #     nn.Conv2d(hidden_features, hidden_features, kernel_size=(5, 1),padding=(5//2, 0),
        #               groups=hidden_features,stride=1),
        #     nn.Conv2d(hidden_features, hidden_features, kernel_size=(1, 5), padding=(0, 5 // 2),
        #               groups=hidden_features, stride=1)
        # )

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False):
        super().__init__()
        ffn_expansion_factor=2
        bias=False
        hidden_features = int(dim * ffn_expansion_factor)
        self.fc1 = nn.Linear(dim, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x) #这里这里
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#########################################################################################

class ConvFFN(nn.Module):

    def __init__(self, dim, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        ffn_expansion_factor = 2
        bias = False
        hidden_features = int(dim * ffn_expansion_factor)
        self.fc1 = nn.Linear(dim, hidden_features)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x,x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x,x_size)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
#####################################################################################
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x
class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

########################################################################################################################
class ElementScale(nn.Module):
    """A learnable element-wise scaler."""

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale
#######################################################################################################################

########################################################################################################################
def ConvL1(in_channels, out_channels, kernel_size, stride,padding, dilation=[1, 1], groups=1):
    padding = [(dilation[0] * (kernel_size[0] - 1)) // 2, (dilation[1] * (kernel_size[1] - 1)) // 2]
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, bias=False),

    )

def ConvL2(in_channels, out_channels, kernel_size, stride,padding, dilation=[1, 1], groups=1):
    padding = [(dilation[0] * (kernel_size[0] - 1)) // 2, (dilation[1] * (kernel_size[1] - 1)) // 2]
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, groups=groups, bias=False),

    )
def Conv1x1(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=False),
        # nn.GELU()
    )


class GMIB(nn.Module):
    #
    def __init__(self,in_ch, out_ch, kernel=1, stride=1, padding=None,dilation=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super(GMIB, self).__init__()

        self.conv1= nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=5,stride=1,padding=2,groups=in_ch)
        ########################ERAM#########################################################################################

        self.con_l1 = ConvL1(in_channels=in_ch, out_channels=out_ch, kernel_size=[7, 1], stride=1,
                               padding=[1, 0], groups=in_ch)

        self.con_l2 = ConvL2(in_channels=in_ch, out_channels=out_ch, kernel_size=[1, 7], stride=1,
                                 padding=[0, 1], groups=in_ch)

        self.con_r1 = ConvL1(in_channels=in_ch, out_channels=out_ch, kernel_size=[9, 1], stride=1,
                               dilation=[dilation, 1], padding=[dilation, 0], groups=in_ch)

        self.con_r2 = ConvL2(in_channels=in_ch, out_channels=out_ch, kernel_size=[1, 9], stride=1,
                                 dilation=[1, dilation], padding=[0, dilation], groups=in_ch)

        #####################################################################################################################
        # self.lka= LKAT(in_ch)
        self.conv = Conv1x1(in_channels=out_ch*2, out_channels=out_ch)



    def forward(self, x):
        x_input = self.conv1(x)
        x1 = self.con_l1(x_input)
        x2 = self.con_r1(x_input)
        l_1 = x1 + x2
        xl_2 = self.con_l2(l_1)
        xr_2 = self.con_r2(l_1)
        out = torch.cat((xl_2, xr_2), 1)
        # out = xl_2+xr_2
        out_b =self.conv(out)
        # out_a = self.lka(out_b)
        return out_b+x

##########################

class SML(nn.Module):
    def __init__(self,
                 embed_dims,
                 attn_dw_dilation=1,
                 kernel=5,
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(SML, self).__init__()

        self.embed_dims = embed_dims
        self.attn_force_fp32 = attn_force_fp32
        self.proj_1 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)
        self.gate = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        self.value1= GMIB(embed_dims, embed_dims, kernel, 1, 1, dilation=attn_dw_dilation)
        self.proj_2 = nn.Conv2d(
            in_channels=embed_dims, out_channels=embed_dims, kernel_size=1)

        # activation for gating and value
        self.act_value = nn.SiLU()
        self.act_gate = nn.SiLU()

        # decompose
        self.sigma = ElementScale(
            embed_dims, init_value=1e-5, requires_grad=True)

    def feat_decompose(self, x):
        x = self.proj_1(x)
        # x_d: [B, C, H, W] -> [B, C, 1, 1]
        x_d = F.adaptive_avg_pool2d(x, output_size=1)
        x = x + self.sigma(x - x_d)
        x = self.act_value(x)
        return x

    def forward_gating(self, g, v):
        with torch.autocast(device_type='cuda', enabled=False):
            g = g.to(torch.float32)
            v = v.to(torch.float32)
            return self.proj_2(self.act_gate(g) * self.act_value(v))

    def forward(self, x):
        shortcut = x.clone()
        # proj 1x1
        x = self.feat_decompose(x)
        # gating and value branch
        g = self.gate(x)
        v = self.value1(x)
        # aggregation
        if not self.attn_force_fp32:
            x = self.proj_2(self.act_gate(g) * self.act_gate(v))
        else:
            x = self.forward_gating(self.act_gate(g), self.act_gate(v))
        x = x + shortcut
        return x
##########################################################################################################################

#########################################################################################################################
class GMIM(nn.Module):
    def __init__(self,
                 embed_dims,
                 ffn_ratio=4.,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 act_type='GELU',
                 norm_type='BN',
                 init_value=1e-5,
                 attn_dw_dilation=1,
                 kernel=5,
                 attn_act_type='SiLU',
                 attn_force_fp32=False,
                ):
        super(GMIM, self).__init__()
        self.out_channels = embed_dims

        # spatial attention
        self.attn = SML(
            embed_dims,
            attn_dw_dilation=attn_dw_dilation,
            kernel=kernel,
            attn_act_type=attn_act_type,
            attn_force_fp32=attn_force_fp32,
        )
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0. else nn.Identity()

        layer_norm_type='WithBias'
        self.norm2 = LayerNorm(embed_dims,layer_norm_type)

        # init layer scale
        self.layer_scale_1 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)), requires_grad=True)
        self.ffn = FeedForward(embed_dims)

    def forward(self, x):
        # spatial
        identity = x
        x = self.layer_scale_1 * self.attn(self.norm2(x))
        x = identity + self.drop_path(x)
        # channel
        identity = x
        x = self.layer_scale_2 * self.ffn(self.norm2(x))
        x = identity + self.drop_path(x)
        return x


@ARCH_REGISTRY.register()
class GMIN_L(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64, num_block=9, num_out_ch=3, upscale=4,p=0.25):
        super(GMIN_L, self).__init__()

        self.conv = nn.Conv2d

        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1,1)

        self.B1 = GMIM(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value= 1e-5,attn_dw_dilation = 1,kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B2 = GMIM(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value= 1e-5,attn_dw_dilation=3,kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B3 = GMIM(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value= 1e-5,attn_dw_dilation=3,kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B4 = GMIM(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value= 1e-5,attn_dw_dilation=5,kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B5 = GMIM(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value= 1e-5,attn_dw_dilation=7,kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B6 = GMIM(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value=1e-5,attn_dw_dilation=7,kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B7 = GMIM(embed_dims=num_feat, ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value=1e-5,attn_dw_dilation=9, kernel=5,attn_act_type='SiLU',attn_force_fp32=False)
        self.B8 = GMIM(embed_dims=num_feat, ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value=1e-5,attn_dw_dilation=13, kernel=5, attn_act_type='SiLU',attn_force_fp32=False)
        self.B9 = GMIM(embed_dims=num_feat, ffn_ratio=4, drop_rate=0.1, drop_path_rate=0, act_type='GELU',norm_type='BN', init_value=1e-5, attn_dw_dilation=13, kernel=5, attn_act_type='SiLU',attn_force_fp32=False)
        # self.B10 = MogaBlock(embed_dims=num_feat, ffn_ratio=4, drop_rate=0.1, drop_path_rate=0, act_type='GELU',norm_type='BN', init_value=1e-5, attn_dw_dilation=13, kernel=5, attn_act_type='SiLU',attn_force_fp32=False)
        # self.B6 = MogaBlock(embed_dims=num_feat,ffn_ratio=4,drop_rate=0.1,drop_path_rate=0,act_type='GELU',norm_type='BN',init_value= 1e-5,attn_dw_dilation=[1,2,3],attn_channel_split=[1,3,4],attn_act_type='SiLU',attn_force_fp32=False)

        self.c1 = nn.Conv2d(num_feat*num_block, num_feat, 1)
        self.GELU = nn.GELU()

        self.c2 = self.conv(num_feat, num_feat, 3, 1,1)
        # self.pconv =  Partial_conv3(in_channels=64, n_div=4, forward='split_cat')


        # self.upsampler = Upsamplers.PixelShuffleDirect(scale=upscale, num_feat=num_feat, num_out_ch=num_out_ch)
         # self.esa = ESA(num_feat)
        self.atten_lk = LKAT(num_feat)
        self.output = nn.Conv2d(num_feat,num_out_ch,kernel_size=3, stride=1, padding=1)
    def forward(self, input):
        self.visual_features = []
        # input = torch.cat([input, input, input, input], dim=1)
        out_fea = self.fea_conv(input)
        out_B1 = self.B1(out_fea)
        out_B2 = self.B2(out_B1)
        out_B3 = self.B3(out_B2)
        out_B4 = self.B4(out_B3)
        out_B5 = self.B5(out_B4)
        out_B6 = self.B6(out_B5)
        out_B7 = self.B7(out_B6)
        out_B8 = self.B8(out_B7)
        out_B9 = self.B9(out_B8)
        # out_B10 = self.B10(out_B9)
        # out_B6 = self.B5(out_B5)
        trunk = torch.cat([out_B1, out_B2, out_B3, out_B4,out_B5,out_B6,out_B7,out_B8,out_B9], dim=1)
        out_B = self.c1(trunk)
        out_B = self.GELU(out_B)
        out_B = self.atten_lk(out_B)
        out_lr = self.c2(out_B) + out_fea
        output = self.output(out_lr)

        # output = self.upsampler(out_lr)
        self.visual_features.extend([output,out_B,out_lr,out_B1,out_B2,out_B3,out_B4,out_fea])
        return output

