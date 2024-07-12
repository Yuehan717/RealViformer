import torch
from torch import nn as nn
from torch.nn import functional as F

from .arch_util import ResidualBlockNoBN, flow_warp, make_layer, LayerNorm, ChannelAttentionBlock
from .spynet_arch import SpyNet
from einops import rearrange
import numbers
##########################################################################
## Layer Norm

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
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, kernel_size=3, stride=1, padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x



##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias, withmask=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.withmask = withmask
        if withmask:
            self.max_dsc = nn.AdaptiveMaxPool1d(1)
            self.avg_dsc = nn.AdaptiveAvgPool1d(1)
            self.linear1 = nn.Linear(2, 1, bias=bias)
            self.linear2 = nn.Linear(dim//num_heads, dim//num_heads, bias=bias)

    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
    
            
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature # b, h, c, c
        
        if self.withmask:
            mask_in = attn.clone()
            b_, h_, c_, _ = mask_in.shape 
            dsc = torch.cat([self.max_dsc(mask_in.reshape(b_, h_*c_, c_)), self.avg_dsc(mask_in.reshape((b_, h_*c_, c_)))], dim=-1).reshape(b_, h_, c_, 2) # b,h,c,2
            mask = self.linear1(dsc) # b,h,c,1
            mask = self.linear2(F.gelu(mask).transpose(-2,-1)).transpose(-2,-1) # b, h, c, 1
            mask = F.sigmoid(mask)
        
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        
        if self.withmask:
            out = out * mask
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out



##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type, 
                 masked=False, squeeze_factor=1.0, ch_compress=False):
        super(TransformerBlock, self).__init__()
        if ch_compress:
            self.norm1 = LayerNorm(dim//squeeze_factor, LayerNorm_type)
            self.attn = Attention(dim//squeeze_factor, num_heads, bias, masked)
        else:
            self.norm1 = LayerNorm(dim, LayerNorm_type)
            self.attn = Attention(dim, num_heads, bias, masked)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.ch_compress = ch_compress and (squeeze_factor > 1)
        if self.ch_compress:
            self.compress = nn.Conv2d(dim, dim//squeeze_factor, kernel_size=3, stride=1, padding=1, bias=bias)
            self.expand = nn.Conv2d(dim//squeeze_factor, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        if self.ch_compress:
            x_compressed = self.compress(x)
            x = x +  self.expand(self.attn(self.norm1(x_compressed)))
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class RealViformer(nn.Module):
    """A recurrent network for video SR. Now only x4 is supported.

    Args:
        num_feat (int): Number of channels. Default: 64.
        num_block (int): Number of residual blocks for each branch. Default: 15
        spynet_path (str): Path to the pretrained weights of SPyNet. Default: None.
    """

    def __init__(self, num_feat=64, num_blocks=[], spynet_path=None, heads=[], ffn_expansion_factor=2, masked=False,
                 merge_head=1, merge_compress=False, merge_compress_factor=1.0,
                 bias=False, LayerNorm_type='BiasFree', ch_compress=False, squeeze_factor=[1,1,1]):
        super().__init__()
        # embedding
        self.num_feat = num_feat
        self.shallow_extraction = nn.Sequential(nn.Conv2d(3, num_feat, 3, 1, 1),
                                                TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=False, ch_compress=False, squeeze_factor=squeeze_factor[0]))
        
        # alignment
        self.spynet = SpyNet(spynet_path)
        self.attn_merge = ChannelAttentionBlock(dim_q= num_feat, dim_kv = num_feat, dim_out=num_feat, num_heads=merge_head, bias=False,
                                                LayerNorm_type='BiasFree', reduction=True, 
                                                ch_compress=merge_compress, squeeze_factor=merge_compress_factor)
        
        # propagation
        # self.forward_trunk = ConvResidualBlocks(num_feat * 2, num_feat, num_block-2)
        ## dim = 48, num_heads = 1, num_block = 4
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=num_feat, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[0])])
        ## dim=96, num_heads = 2, num_block=6
        self.down1_2 = Downsample(num_feat) ## From Level 1 to Level 2
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[1])])
        ## dim=192, num_heads = 4, num_block=6
        self.down2_3 = Downsample(int(num_feat*2**1)) ## From Level 2 to Level 3
        self.latent= nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[2])])
        ## dim=96, num_heads = 2, num_block=6
        self.up3_2 = Upsample(int(num_feat*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(num_feat*2**2), int(num_feat*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[1])])
        ## dim=96, num_heads = 1, num_block=4
        self.up2_1 = Upsample(int(num_feat*2**1))  ## From Level 2 to Level 1  (NO 1x1 conv to reduce channels)
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[0])])
        ## dim=96, num_heads = 1, num_block=2
        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(num_feat*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, masked=masked,ch_compress=ch_compress, squeeze_factor=squeeze_factor[0]) for i in range(num_blocks[3])])
        
        # reconstruction
        self.compress = nn.Conv2d(num_feat*2**1, num_feat, kernel_size=1, bias=True)
        # self.fusion = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(num_feat*2**1, num_feat * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(num_feat, 64 * 4, 3, 1, 1, bias=True)
        self.conv_hr = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1)

        self.pixel_shuffle = nn.PixelShuffle(2)

        # activation functions
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def get_flow(self, x):
        
        b, n, c, h, w = x.size()

        x_1 = x[:, :-1, :, :, :].reshape(-1, c, h, w)
        x_2 = x[:, 1:, :, :, :].reshape(-1, c, h, w)
        
        # print(x_1.size(), x_2.size())

        flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        # flows_forward = self.spynet(x_2, x_1).view(b, n - 1, 2, h, w)
        
        return flows_forward
     
    # @torch.compile
    def forward(self, x, current_iter=None):
        """Forward function of BasicVSR.

        Args:
            x: Input frames with shape (b, n, c, h, w). n is the temporal dimension / number of frames.
        """
        b, n, _, h, w = x.size()
        # self.check_if_mirror_extended(x)
        flows_forward = self.get_flow(x.clone())
        
        out_l = []
        # forward branch
        # feat_prop = torch.zeros_like(feat_prop)
        feat_prop = x.new_zeros(b, self.num_feat, h, w)
        for i in range(0, n):
            x_i = x[:, i, :, :, :]
            feat_shallow = self.shallow_extraction(x_i)
            if i > 0:
                flow = flows_forward[:, i - 1, :, :, :]
                feat_prop = flow_warp(feat_prop, flow.permute(0, 2, 3, 1))
            ## alignment ##
            feat_prop = self.attn_merge(feat_shallow, feat_prop)
            # if hasattr(self, 'align_correction'):
            #     feat_prop = self.align_correction(feat_shallow, feat_prop)            
            
            ## Propagation ##
            out_enc_level1 = self.encoder_level1(feat_prop)
        
            inp_enc_level2 = self.down1_2(out_enc_level1)
            out_enc_level2 = self.encoder_level2(inp_enc_level2)

            inp_enc_level3 = self.down2_3(out_enc_level2)     
            latent = self.latent(inp_enc_level3) 

            inp_dec_level2 = self.up3_2(latent)
            inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
            inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
            out_dec_level2 = self.decoder_level2(inp_dec_level2) 

            inp_dec_level1 = self.up2_1(out_dec_level2)
            inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
            out_dec_level1 = self.decoder_level1(inp_dec_level1)
            
            out = self.refinement(out_dec_level1)
            feat_prop = self.compress(out)
            # upsample
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
            out = self.lrelu(self.conv_hr(out))
            # base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out = self.conv_last(out)
            base = F.interpolate(x_i, scale_factor=4, mode='bilinear', align_corners=False)
            out += base
            
            out_l.append(out)

        return torch.stack(out_l, dim=1)

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

