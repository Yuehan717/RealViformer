import collections.abc
import math
import torch
import warnings
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from einops import rearrange
import numbers


@torch.no_grad() # @ is a decorator, which takes the decorated function as the argument of the function after @
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2 current position 
    grid.requires_grad = False

    vgrid = grid + flow # for all n batches do the same  current position --> warped position
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode) #, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)



def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class ConvResidualBlocks(nn.Module):
    """Conv and residual block used in BasicVSR.

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True),
            make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)
    
class ResidualBlocks(nn.Module):
    """Residual block without shallow conv

    Args:
        num_in_ch (int): Number of input channels. Default: 3.
        num_out_ch (int): Number of output channels. Default: 64.
        num_block (int): Number of residual blocks. Default: 15.
    """

    def __init__(self, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = make_layer(ResidualBlockNoBN, num_block, num_feat=num_out_ch)

    def forward(self, fea):
        return self.main(fea)

class Conv2dwithActication(nn.Module):
    """ Conv2d layers with activation
    """
    def __init__(self, num_in_ch, num_out_ch, num_layer=1):
        super().__init__()
        main = [nn.Conv2d(num_in_ch, num_out_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True)]
        for _ in range(num_layer-1):
            main.append(nn.Conv2d(num_out_ch, num_out_ch, 3, 1, 1, bias=True))
            main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.main = nn.Sequential(*main)
        
    def forward(self, fea):
        return self.main(fea)

##########################################################################
## LayerNorm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
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
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim_q, dim_k, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim_k, dim_k*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim_k*2, dim_k*2, kernel_size=3, stride=1, padding=1, groups=dim_k*2, bias=bias)
        self.q = nn.Conv2d(dim_q, dim_q, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim_q, dim_q, kernel_size=1, bias=bias)
        


    def forward(self, x, y):
        b,c_kv,h,w = x.shape

        kv = self.kv_dwconv(self.kv(x))
        k,v = kv.chunk(2, dim=1)
        
        # c_q = y.size(1)
        q = self.q(y)
           
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # c_q, c_kv (64, 32)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) # (b, 64, h, w)
        return out


##########################################################################
# Attention block with layer norm
class AttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads, bias, LayerNorm_type=None, reduction='sum'):
        super(AttentionBlock, self).__init__()
        if LayerNorm_type is not None:
            self.norm_q = LayerNorm(dim_q, LayerNorm_type)
            self.norm_kv = LayerNorm(dim_kv, LayerNorm_type)
        self.attn = Attention(dim_q, dim_kv, num_heads, bias)
        if reduction == 'cat':
            self.fusion = nn.Conv2d(dim_q*2, dim_q, kernel_size=1, bias=bias)
    
    def forward(self, x, y):
        if hasattr(self, 'norm_kv'):
            x = self.norm_kv(x)
        if hasattr(self, 'norm_q'):
            y = self.norm_q(y)
        
        v = self.attn(x, y) ## ()
        
        if hasattr(self, 'fusion'):
            return self.fusion(torch.cat((y, v), dim=1))
        else:
            return y + v
        

##########################################################################
## Simple attention function (parameter-free)
    
def cross_attention(x, ref):
    b,c0,h,w = x.shape
    _,c1,_,_ = ref.shape
    q,k,v = x.clone(), ref.clone(), ref.clone()   
    
    q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=1)
    k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=1)
    v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=1)

    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)

    attn = (q @ k.transpose(-2, -1)) * 1 # self.temperature [c0, c1]
    max_, _ = torch.max(attn, dim=-1, keepdim=True)
    attn_submax = attn - max_
    attn = attn.softmax(dim=-1)
    attn = attn * torch.lt(attn_submax, 0).int()
    attn = attn.softmax(dim=-1)

    out = (attn @ v) # [c0, hw]
    
    out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=1, h=h, w=w)

    # out = self.project_out(out)
    return out


##########################################################################
## Multi-DConv Head Cross-Attention (MDCA)
class CrossChannelAttention(nn.Module):
    def __init__(self, dim_q, dim_k, num_heads, bias):
        super(CrossChannelAttention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim_k, dim_k*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim_k*2, dim_k*2, kernel_size=3, stride=1, padding=1, groups=dim_k*2, bias=bias)
        self.q = nn.Conv2d(dim_q, dim_q, kernel_size=3, stride=1, padding=1, bias=bias)
        self.project_out = nn.Conv2d(dim_q, dim_q, kernel_size=1, bias=bias)
        
    def forward(self, x, y, return_attn=False):
        b,c_kv,h,w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k,v = kv.chunk(2, dim=1)
        
        # c_q = y.size(1)
        q = self.q(x)
           
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v)

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out) # (b, 64, h, w)
        if return_attn:
            return out, attn
        else:
            return out
    
##########################################################################
# Channel-wise Attention block with layer norm
class ChannelAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, dim_out,num_heads, bias, LayerNorm_type=None, 
                 reduction=True, ch_compress=False, squeeze_factor=1):
        super(ChannelAttentionBlock, self).__init__()
        self.ch_compress = ch_compress and (squeeze_factor > 1)
        if self.ch_compress:
            if LayerNorm_type is not None:
                self.norm_q = LayerNorm(dim_q//squeeze_factor, LayerNorm_type)
                self.norm_kv = LayerNorm(dim_kv//squeeze_factor, LayerNorm_type)
            self.attn = CrossChannelAttention(dim_q//squeeze_factor, dim_kv//squeeze_factor, num_heads, bias)
        else:
            if LayerNorm_type is not None:
                self.norm_q = LayerNorm(dim_q, LayerNorm_type)
                self.norm_kv = LayerNorm(dim_kv, LayerNorm_type)
            self.attn = CrossChannelAttention(dim_q, dim_kv, num_heads, bias)
        self.reduction = reduction
        if reduction:
            self.norm_out = LayerNorm(dim_kv, LayerNorm_type)
            self.ffn = nn.Sequential(nn.Conv2d(dim_kv+dim_q, dim_out, kernel_size=1, bias=bias),
                                    nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, groups=dim_out, bias=bias),
                                    nn.GELU(),
                                    nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=bias))
        
        if self.ch_compress:
            self.compress_q = nn.Conv2d(dim_q, dim_q//squeeze_factor, kernel_size=3, stride=1, padding=1, bias=bias)
            self.compress_kv = nn.Conv2d(dim_kv, dim_kv//squeeze_factor, kernel_size=3, stride=1, padding=1, bias=bias)
            self.expand = nn.Conv2d(dim_q//squeeze_factor, dim_q, kernel_size=3, stride=1, padding=1, bias=bias)
    
    def forward(self, x, y, return_attn=False):
        
        if self.ch_compress:
            x_compressed = self.compress_q(x)
            y_compressed = self.compress_kv(y)
            if hasattr(self, 'norm_kv'):
                y_compressed = self.norm_kv(y_compressed)
            if hasattr(self, 'norm_q'):
                x_compressed = self.norm_q(x_compressed)
            v = self.expand(self.attn(x_compressed, y_compressed, return_attn))
        else:
            if hasattr(self, 'norm_kv'):
                y = self.norm_kv(y)
            if hasattr(self, 'norm_q'):
                x = self.norm_q(x)
            v = self.attn(x, y, return_attn) ## ()
        if self.reduction:
            out = self.ffn(torch.cat([x,self.norm_out(v)], dim=1))
        return out
