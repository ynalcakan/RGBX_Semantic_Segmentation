import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import FeatureRectifyModule as FRM
from ..net_utils import ImprovedFeatureRectifyModule as IFRM
from ..net_utils import ImprovedFeatureFusionModule as IFFM
import math
import time
from engine.logger import get_logger
from config import config

logger = get_logger()

bn_config = {}
bn_config['SyncBN_MOM'] = 3e-4
bn_config['BN_MOM'] = 0.9
bn_config['norm_typ'] = 'batch_norm'
from torch.nn import SyncBatchNorm as SynchronizedBatchNorm2d

norm_layer = partial(SynchronizedBatchNorm2d, momentum=float(bn_config['SyncBN_MOM']))

class myLayerNorm(nn.Module):
    def __init__(self, inChannels):
        super().__init__()
        self.norm == nn.LayerNorm(inChannels, eps=1e-5)

    def forward(self, x):
        # reshaping only to apply Layer Normalization layer
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2) # B*C*H*W -> B*C*HW -> B*HW*C
        x = self.norm(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() # B*HW*C -> B*H*W*C -> B*C*H*W

        return x


class NormLayer(nn.Module):
    def __init__(self, inChannels, norm_type=bn_config['norm_typ']):
        super().__init__()
        self.inChannels = inChannels
        self.norm_type = norm_type
        if norm_type == 'batch_norm':
            # print('Adding Batch Norm layer') # for testing
            self.norm = nn.BatchNorm2d(inChannels, eps=1e-5, momentum=float(bn_config['BN_MOM']))
        elif norm_type == 'sync_bn':
            # print('Adding Sync-Batch Norm layer') # for testing
            self.norm = norm_layer(inChannels)
        elif norm_type == 'layer_norm':
            # print('Adding Layer Norm layer') # for testing
            self.norm = myLayerNorm(inChannels)
        else:
            raise NotImplementedError

    def forward(self, x):

        x = self.norm(x)
        
        return x
    
    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, norm_type={self.norm_type})'

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class LayerScale(nn.Module):
    '''
    Layer scale module.
    References:
      - https://arxiv.org/abs/2103.17239
    '''
    def __init__(self, inChannels, init_value=1e-2):
        super().__init__()
        self.inChannels = inChannels
        self.init_value = init_value
        self.layer_scale = nn.Parameter(init_value * torch.ones((inChannels)), requires_grad=True)

    def forward(self, x):
        if self.init_value == 0.0:
            return x
        else:
            scale = self.layer_scale.unsqueeze(-1).unsqueeze(-1) # C, -> C,1,1
            return scale * x

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={self.inChannels}, init_value={self.init_value})'

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
def stochastic_depth(input: torch.Tensor, p: float,
                     mode: str, training: bool =  True):
    
    if not training or p == 0.0:
        # print(f'not adding stochastic depth of: {p}')
        return input
    
    survival_rate = 1.0 - p
    if mode == 'row':
        shape = [input.shape[0]] + [1] * (input.ndim - 1) # just converts BXCXHXW -> [B,1,1,1] list
    elif mode == 'batch':
        shape = [1] * input.ndim

    noise = torch.empty(shape, dtype=input.dtype, device=input.device)
    noise = noise.bernoulli_(survival_rate)
    if survival_rate > 0.0:
        noise.div_(survival_rate)
    # print(f'added sDepth of: {p}')
    return input * noise

class StochasticDepth(nn.Module):
    '''
    Stochastic Depth module.
    It performs ROW-wise dropping rather than sample-wise. 
    mode (str): ``"batch"`` or ``"row"``.
                ``"batch"`` randomly zeroes the entire input, ``"row"`` zeroes
                randomly selected rows from the batch.
    References:
      - https://pytorch.org/vision/stable/_modules/torchvision/ops/stochastic_depth.html#stochastic_depth
    '''
    def __init__(self, p=0.5, mode='row'):
        super().__init__()
        self.p = p
        self.mode = mode
    
    def forward(self, input):
        return stochastic_depth(input, self.p, self.mode, self.training)
    
    def __repr__(self):
       s = f"{self.__class__.__name__}(p={self.p})"
       return s

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
def resize(input,
           size=None,
           scale_factor=None,
           mode='bilinear',
           align_corners=None,
           warning=True):

    return F.interpolate(input, size, scale_factor, mode, align_corners)


class DownSample(nn.Module):
    def __init__(self, in_channels, embed_dim, stride=2, kernelSize=3):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(kernelSize, kernelSize),
                              stride=stride, padding=(kernelSize//2, kernelSize//2))
        # stride 4 => 4x down sample
        # stride 2 => 2x down sample
    def forward(self, x):
        x = self.proj(x)
        # B, C, H, W = x.size()
        # x = x.flatten(2).transpose(1,2)
        return x
    
#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class DWConv3x3(nn.Module):
    '''Depth wise conv'''
    def __init__(self, dim=768):
        super(DWConv3x3, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

class ConvBNRelu(nn.Module):

    @classmethod
    def _same_paddings(cls, kernel):
        if kernel == 1:
            return 0
        elif kernel == 3:
            return 1

    def __init__(self, inChannels, outChannels, kernel=3, stride=1, padding='same',
                 dilation=1, groups=1):
        super().__init__()

        if padding == 'same':
            padding = self._same_paddings(kernel)
        
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel,
                              padding=padding, stride=stride, dilation=dilation,
                              groups=groups, bias=False)
        self.norm = NormLayer(outChannels, norm_type=bn_config['norm_typ'])
        self.act = nn.ReLU(inplace=True)
    
    def forward(self, x):
        
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x

class SeprableConv2d(nn.Module):
    def __init__(self, inChannels, outChannels, kernal_size=3, bias=False):
        self.dwconv = nn.Conv2d(inChannels, inChannels, kernal_size=kernal_size,
                                groups=inChannels, bias=bias)
        self.pwconv = nn.Conv2d(inChannels, inChannels, kernal_size=1, bias=bias)

    def forward(self, x):

        x = self.dwconv(x)
        x = self.pwconv(x)
        
        return x

class ConvRelu(nn.Module):
    def __init__(self, inChannels, outChannels, kernel=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(inChannels, outChannels, kernel_size=kernel, bias=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.conv(x)
        x = self.act(x)
        
        return x

class MSCA(nn.Module):
    def __init__(self, dim):
        super(MSCA, self).__init__()
        self.conv55 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim) 
        self.conv17_0 = nn.Conv2d(dim, dim, (1,7), padding=(0, 3), groups=dim)
        self.conv17_1 = nn.Conv2d(dim, dim, (7,1), padding=(3, 0), groups=dim)
        self.conv111_0 = nn.Conv2d(dim, dim, (1,11), padding=(0,5), groups=dim)
        self.conv111_1 = nn.Conv2d(dim, dim, (11,1), padding=(5,0), groups=dim)
        self.conv211_0 = nn.Conv2d(dim, dim, (1,21), padding=(0, 10), groups=dim)
        self.conv211_1 = nn.Conv2d(dim, dim, (21,1), padding=(10, 0), groups=dim)
        self.conv11 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        skip = x.clone()
        c55 = self.conv55(x)
        c17 = self.conv17_1(self.conv17_0(x))
        c111 = self.conv111_1(self.conv111_0(x))
        c211 = self.conv211_1(self.conv211_0(x))
        add = c55 + c17 + c111 + c211
        mixer = self.conv11(add)
        return mixer * skip

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, hid_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hid_channels, 1)
        self.dwconv = DWConv3x3(hid_channels)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hid_channels, out_channels, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, ffn_ratio=4., ls_init_val=1e-2, drop_path=0.):
        super().__init__()
        self.norm1 = NormLayer(dim)
        self.attn = MSCA(dim)
        self.ls1 = LayerScale(dim, ls_init_val)
        self.drop_path1 = StochasticDepth(drop_path)

        self.norm2 = NormLayer(dim)
        self.ffn = FFN(dim, dim, int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim, ls_init_val)
        self.drop_path2 = StochasticDepth(drop_path)

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x

class SegNextEncoder(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., ls_init_val=1e-6, out_indices=(0, 1, 2, 3)):
        super().__init__()
        self.depths = depths
        self.out_indices = out_indices
        self.rectify_module = config.rectify_module
        self.fusion_module = config.fusion_module

        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            NormLayer(dims[0])
        )
        self.extra_stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            NormLayer(dims[0])
        )

        self.stages = nn.ModuleList()
        self.extra_stages = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], ffn_ratio=4, ls_init_val=ls_init_val, drop_path=dpr[cur + j])
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            self.extra_stages.append(stage)
            cur += depths[i]

            if i < 3:
                self.stages.append(DownSample(dims[i], dims[i+1], 2))
                self.extra_stages.append(DownSample(dims[i], dims[i+1], 2))

        # Initialize rectify modules
        if self.rectify_module == 'FRM':    
            self.FRMs = nn.ModuleList([
                FRM(dim=dims[i], reduction=1) for i in range(4)
            ])
        elif self.rectify_module == 'IFRM':
            self.FRMs = nn.ModuleList([
                IFRM(dim=dims[i], reduction=1) for i in range(4)
            ])
        else:
            raise ValueError(f"Invalid rectify_module: {self.rectify_module}. Must be 'FRM' or 'IFRM'")

        # Initialize fusion modules
        if self.fusion_module == 'FFM':
            self.FFMs = nn.ModuleList([
                FFM(dim=dims[i], reduction=1, num_heads=8) for i in range(4)
            ])
        elif self.fusion_module == 'IFFM':
            self.FFMs = nn.ModuleList([
                IFFM(dim=dims[i], reduction=1, num_heads=8) for i in range(4)
            ])
        else:
            raise ValueError(f"Invalid fusion_module: {self.fusion_module}. Must be 'FFM' or 'IFFM'")

        self.norm = nn.ModuleList([NormLayer(dims[i]) for i in range(4)])
        self.extra_norm = nn.ModuleList([NormLayer(dims[i]) for i in range(4)])

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger.info(f'Loading pretrained weights from {pretrained}')
            load_dualpath_model(self, pretrained)
        elif pretrained is None:
            logger.info('No pretrained weights, using random initialization')
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x_rgb, x_e):
        x_rgb = self.stem(x_rgb)
        x_e = self.extra_stem(x_e)
        outs = []
        
        # Process each stage separately
        stage_outputs = []
        extra_stage_outputs = []
        
        # Stage 1
        x_rgb = self.stages[0](x_rgb)
        x_e = self.extra_stages[0](x_e)
        stage_outputs.append(x_rgb)
        extra_stage_outputs.append(x_e)
        
        # Downsample after stage 1
        x_rgb = self.stages[1](x_rgb)  # This is a DownSample layer
        x_e = self.extra_stages[1](x_e)
        
        # Stage 2
        x_rgb = self.stages[2](x_rgb)
        x_e = self.extra_stages[2](x_e)
        stage_outputs.append(x_rgb)
        extra_stage_outputs.append(x_e)
        
        # Downsample after stage 2
        x_rgb = self.stages[3](x_rgb)  # This is a DownSample layer
        x_e = self.extra_stages[3](x_e)
        
        # Stage 3
        x_rgb = self.stages[4](x_rgb)
        x_e = self.extra_stages[4](x_e)
        stage_outputs.append(x_rgb)
        extra_stage_outputs.append(x_e)
        
        # Downsample after stage 3
        x_rgb = self.stages[5](x_rgb)  # This is a DownSample layer
        x_e = self.extra_stages[5](x_e)
        
        # Stage 4
        x_rgb = self.stages[6](x_rgb)
        x_e = self.extra_stages[6](x_e)
        stage_outputs.append(x_rgb)
        extra_stage_outputs.append(x_e)
        
        # Process output stages
        for i in range(4):
            norm_layer = self.norm[i]
            extra_norm_layer = self.extra_norm[i]
            x_rgb_out = norm_layer(stage_outputs[i])
            x_e_out = extra_norm_layer(extra_stage_outputs[i])
            
            x_rgb_out, x_e_out = self.FRMs[i](x_rgb_out, x_e_out)
            x_fused = self.FFMs[i](x_rgb_out, x_e_out)
            outs.append(x_fused)
            
        return outs

    def forward(self, x_rgb, x_e):
        return self.forward_features(x_rgb, x_e)

def load_dualpath_model(model, model_file):
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.find('stem') >= 0:
            state_dict[k] = v
            state_dict[k.replace('stem', 'extra_stem')] = v
        elif k.find('stages') >= 0:
            state_dict[k] = v
            state_dict[k.replace('stages', 'extra_stages')] = v
        elif k.find('norm') >= 0:
            state_dict[k] = v
            state_dict[k.replace('norm', 'extra_norm')] = v

    t_ioend = time.time()

    model.load_state_dict(state_dict, strict=False)
    del state_dict
    
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

class segnext_tiny(SegNextEncoder):
    def __init__(self, **kwargs):
        super(segnext_tiny, self).__init__(
            depths=[3, 3, 9, 3], dims=[32, 64, 160, 256], **kwargs)

class segnext_small(SegNextEncoder):
    def __init__(self, **kwargs):
        super(segnext_small, self).__init__(
            depths=[3, 3, 27, 3], dims=[64, 128, 320, 512], **kwargs)

class segnext_base(SegNextEncoder):
    def __init__(self, **kwargs):
        super(segnext_base, self).__init__(
            depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)

class segnext_large(SegNextEncoder):
    def __init__(self, **kwargs):
        super(segnext_large, self).__init__(
            depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)

