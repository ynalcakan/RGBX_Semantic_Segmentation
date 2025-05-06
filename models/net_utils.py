import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
import math
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, global_max_pool, SAGPooling
from torch_geometric.utils import dense_to_sparse
from torch_geometric.nn import radius_graph
from config import config as C
# Feature Rectify Module
class ChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
                    nn.Linear(self.dim * 4, self.dim * 4 // reduction),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.dim * 4 // reduction, self.dim * 2), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # Ensure input channels match expected dimensions
        assert C == self.dim, f"Input channel dimension {C} does not match expected dimension {self.dim}"
        
        x = torch.cat((x1, x2), dim=1)  # B, 2*dim, H, W
        avg = self.avg_pool(x).flatten(1)  # B, 2*dim
        max = self.max_pool(x).flatten(1)  # B, 2*dim
        
        y = torch.cat((avg, max), dim=1)  # B, 4*dim
        y = self.mlp(y)  # B, 2*dim
        
        # Reshape to the expected output format
        channel_weights = y.view(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2, B, dim, 1, 1
        return channel_weights
    

class ImprovedChannelWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ImprovedChannelWeights, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim * 4 // reduction),
            nn.LayerNorm(self.dim * 4 // reduction),
            nn.GELU(),
            nn.Linear(self.dim * 4 // reduction, self.dim * 2),
            nn.LayerNorm(self.dim * 2)
        )
        
        self.gate = nn.Sequential(
            nn.Linear(self.dim * 2, self.dim * 2),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        # Ensure input channels match expected dimensions
        assert C == self.dim, f"Input channel dimension {C} does not match expected dimension {self.dim}"
        
        x = torch.cat((x1, x2), dim=1)  # B, 2*dim, H, W
        avg = self.avg_pool(x).flatten(1)  # B, 2*dim
        max = self.max_pool(x).flatten(1)  # B, 2*dim
        
        y = torch.cat((avg, max), dim=1)  # B, 4*dim
        y = self.mlp(y)  # B, 2*dim
        
        # Gating mechanism
        g = self.gate(y)
        y = y * g
        
        # Reshape to the expected output format
        channel_weights = y.view(B, 2, self.dim, 1, 1).permute(1, 0, 2, 3, 4)  # 2, B, dim, 1, 1
        return channel_weights
    

class SpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1, kernel_size=1):
        super(SpatialWeights, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.mlp = nn.Sequential(
                    nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=self.kernel_size),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.dim // reduction, 2, kernel_size=self.kernel_size), 
                    nn.Sigmoid())

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1) # B 2C H W
        spatial_weights = self.mlp(x).reshape(B, 2, 1, H, W).permute(1, 0, 2, 3, 4) # 2 B 1 H W
        return spatial_weights
    

class ImprovedSpatialWeights(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ImprovedSpatialWeights, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv2d(self.dim * 2, self.dim // reduction, kernel_size=1)
        self.norm1 = nn.BatchNorm2d(self.dim // reduction)
        self.conv2 = nn.Conv2d(self.dim // reduction, self.dim // reduction, kernel_size=1)
        self.norm2 = nn.BatchNorm2d(self.dim // reduction)
        self.conv3 = nn.Conv2d(self.dim // reduction, 2, kernel_size=1)

    def forward(self, x1, x2):
        B, _, H, W = x1.shape
        x = torch.cat((x1, x2), dim=1)
        
        # First convolution block
        y = self.conv1(x)
        y = self.norm1(y)
        y = F.gelu(y)
        
        # Residual connection
        residual = y
        
        # Second convolution block
        y = self.conv2(y)
        y = self.norm2(y)
        y = F.gelu(y)
        
        # Add residual
        y = y + residual
        
        # Final convolution to get 2 channels
        y = self.conv3(y)
        # y = torch.sigmoid(y)
        
        spatial_weights = y.view(B, 2, 1, H, W).permute(1, 0, 2, 3, 4)
        return spatial_weights


class FeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1, lambda_c=.5, lambda_s=.5):
        super(FeatureRectifyModule, self).__init__()
        self.lambda_c = lambda_c
        self.lambda_s = lambda_s
        self.channel_weights = ChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = SpatialWeights(dim=dim, reduction=reduction)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
    
    def forward(self, x1, x2):
        # Check dimensions
        B, C, H, W = x1.shape
        assert x1.shape == x2.shape, f"Input shapes do not match: {x1.shape} vs {x2.shape}"
        assert C == self.channel_weights.dim, f"Input channel dimension {C} does not match expected dimension {self.channel_weights.dim}"
        
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        out_x1 = x1 + self.lambda_c * channel_weights[1] * x2 + self.lambda_s * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_c * channel_weights[0] * x1 + self.lambda_s * spatial_weights[0] * x1
        
        return out_x1, out_x2


class ImprovedFeatureRectifyModule(nn.Module):
    def __init__(self, dim, reduction=1):
        super(ImprovedFeatureRectifyModule, self).__init__()
        self.channel_weights = ImprovedChannelWeights(dim=dim, reduction=reduction)
        self.spatial_weights = ImprovedSpatialWeights(dim=dim, reduction=reduction)
        
        # Dynamic lambda
        self.lambda_channel = nn.Parameter(torch.tensor(0.5))
        self.lambda_spatial = nn.Parameter(torch.tensor(0.5))
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        # Check dimensions
        B, C, H, W = x1.shape
        assert x1.shape == x2.shape, f"Input shapes do not match: {x1.shape} vs {x2.shape}"
        assert C == self.channel_weights.dim, f"Input channel dimension {C} does not match expected dimension {self.channel_weights.dim}"
        
        channel_weights = self.channel_weights(x1, x2)
        spatial_weights = self.spatial_weights(x1, x2)
        
        # Use dynamic lambda values
        out_x1 = x1 + self.lambda_channel * channel_weights[1] * x2 + self.lambda_spatial * spatial_weights[1] * x2
        out_x2 = x2 + self.lambda_channel * channel_weights[0] * x1 + self.lambda_spatial * spatial_weights[0] * x1
        
        # Apply layer normalization
        out_x1 = self.norm(out_x1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        out_x2 = self.norm(out_x2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        
        return out_x1, out_x2


# ---------------------  Feature Fusion Module --------------------- #
# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = x1.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        q2 = x2.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        k1, v1 = self.kv1(x1).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k2, v2 = self.kv2(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous() 

        return x1, x2
    
class ImprovedCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ImprovedCrossAttention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q1 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q2 = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = self.q1(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1, v1 = self.kv1(x1).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        q2 = self.q2(x2).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2, v2 = self.kv2(x2).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)

        attn1 = (q1 @ k2.transpose(-2, -1)) * self.scale
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        x1 = (attn1 @ v2).transpose(1, 2).reshape(B, N, C)
        x1 = self.proj1(x1)
        x1 = self.proj_drop(x1)

        attn2 = (q2 @ k1.transpose(-2, -1)) * self.scale
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        x2 = (attn2 @ v1).transpose(1, 2).reshape(B, N, C)
        x2 = self.proj2(x2)
        x2 = self.proj_drop(x2)

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2
    
class ImprovedCrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.cross_attn = ImprovedCrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        residual1, residual2 = x1, x2
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(residual1 + self.end_proj1(y1))
        out_x2 = self.norm2(residual2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
                        nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
                        norm_layer(out_channels) 
                        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out
    
class ImprovedChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(ImprovedChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True),
            nn.Conv2d(out_channels//reduction, out_channels//reduction, kernel_size=3, stride=1, padding=1, bias=True, groups=out_channels//reduction),
            nn.GELU(),
            nn.Conv2d(out_channels//reduction, out_channels, kernel_size=1, bias=True),
            norm_layer(out_channels) 
        )
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out
          
class GraphChannelEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):
        super(GraphChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.graph = GraphConstructor(k=0.5, r=1.0)
        if C.gfm_net_type == 'GCNNetwork':
            self.channel_embed = GCNNetwork(in_channels=in_channels, out_channels=out_channels, reduction=reduction, norm_layer=norm_layer)
        elif C.gfm_net_type == 'GCNNetworkV2':
            self.channel_embed = GCNNetworkV2(in_channels=in_channels, out_channels=out_channels, reduction=reduction, norm_layer=norm_layer)
        elif C.gfm_net_type == 'GCNNetworkV3':
            self.channel_embed = GCNNetworkV3(in_channels=in_channels, out_channels=out_channels, reduction=reduction, norm_layer=norm_layer)
        self.norm = norm_layer(out_channels)
        
    def forward(self, x, H, W):
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        return out

class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        return merge
    

class ImprovedFeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = ImprovedCrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ImprovedChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        return merge
    
class GraphFeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = GraphChannelEmbed(in_channels=dim*2, out_channels=dim, reduction=reduction, norm_layer=norm_layer)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)
        x1, x2 = self.cross(x1, x2) 
        merge = torch.cat((x1, x2), dim=-1)
        merge = self.channel_emb(merge, H, W)
        
        return merge
    
class GraphConstructor(nn.Module):
    def __init__(self, k=0.5, r=1.0):
        super(GraphConstructor, self).__init__()
        self.k = k # Controlling the weight between coordinate similarity and feature similarity.
        self.r = r # The bandwidth for the RBF kernel used in coordinate similarity calculation.

    def rbf_kernel(self, x, y):
        # RBF kernel to measure the similarity between two points
        dist = torch.norm(x - y, p=2, dim=-1)
        return torch.exp(- (dist ** 2) / (2 * self.r ** 2))

    def compute_coordinate_similarity(self, X):
        # Euclidean distance.
        # X: Tensor of shape (H * W, C), where C is the feature dimension for each pixel.
        H, W, C = X.shape
        # C, H, W, _ = X.shape

        nodes = X.view(-1, C)  # Flatten the HxW dimension into nodes of shape (H*W, C)

        Ad = torch.zeros(H * W, H * W)  # Initialize similarity matrix

        # Compute the coordinate similarity between all pairs of nodes
        for i in range(H * W):
            for j in range(i + 1, H * W):
                # Get the coordinates (i, j) in 2D grid space
                si = torch.tensor([i // W, i % W])  # Coordinate of node i
                sj = torch.tensor([j // W, j % W])  # Coordinate of node j

                similarity = self.rbf_kernel(si.float(), sj.float())  # Apply RBF on coordinates
                Ad[i, j] = similarity
                Ad[j, i] = similarity  # Symmetric matrix

        # Normalize the coordinate similarity matrix
        Ad = Ad / Ad.sum(dim=-1, keepdim=True)
        
        return Ad

    def compute_feature_similarity(self, X):
        # Compute the feature similarity As between nodes based on their feature vectors.
        # X: Tensor of shape (H * W, C), where C is the feature dimension for each pixel.
        H, W, C = X.shape
        nodes = X.view(-1, C)  # Flatten the HxW -> (H*W, C)

        As = torch.matmul(nodes, nodes.t())  # Compute pairwise similarity using Hadamard product
        As = F.softmax(As, dim=-1)  # Normalize with Softmax along rows
        
        # Thresholding to remove weak similarities (based on average threshold)
        avg_similarity = As.mean(dim=-1, keepdim=True)
        As[As < avg_similarity] = 0  # Set values below threshold to 0
        
        return As

    def create_graph(self, X):
        # Build a sparse graph using radius neighbors on the HxW grid
        H, W, C = X.shape
        # create grid coordinates for each pixel
        row = torch.arange(H, device=X.device)
        col = torch.arange(W, device=X.device)
        grid_r, grid_c = torch.meshgrid(row, col, indexing='ij')
        coords = torch.stack((grid_r, grid_c), dim=-1).view(-1, 2).float()
        # generate edges where pixel-pairs are within radius self.r
        edge_index = radius_graph(coords, r=self.r, loop=False)
        return edge_index
    
class GCNNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):

        super(GCNNetwork, self).__init__()
        self.preprocess = nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True) # conv1x1 to fuse features
        # GCN layers
         # dynamically create exactly C.GCN_layers-1 hidden layers + 1 output layer
        self.convs = nn.ModuleList()
        for _ in range(C.GCN_layers - 1):
            self.convs.append(GCNConv(out_channels//reduction, out_channels//reduction, aggr='mean', bias=True))
        # final maps to full out_channels
        self.convs.append(GCNConv(out_channels//reduction, out_channels, aggr='mean', bias=True))
        
        self.norm = norm_layer(out_channels) 
        # k=0.5 controls how much the model balances pure spatial (coordinate) similarity vs. feature-based similarity when building its adjacency.
        # r=1.0 is used both as the bandwidth in the RBF (Gaussian) kernel over pixel coordinates and as the search radius when you do neighborhood graph construction.
        # Graph constructor
        self.graph = GraphConstructor(k=0.5, r=1.0)

    def forward(self, x):
        """
        Forward pass: x is a batch of spatial features [B, C_in, H, W].
        We build one graph per sample on the H×W grid, run two GCNConv layers, pool, and normalize.
        """
        B, C_in, H, W = x.shape
        # initial conv projection
        x_reduced = self.preprocess(x)                     # [B, C_r, H, W]

        # build batched edge_index for all B graphs
        edge_indices = []
        for b in range(B):
            # per-sample node features: H×W×C_r
            feat = x_reduced[b].permute(1, 2, 0).contiguous()  # [H, W, C_r]
            A = self.graph.create_graph(feat)
            ei, _ = dense_to_sparse(A)
            ei = ei + b * (H * W)
            edge_indices.append(ei)
        edge_index = torch.cat(edge_indices, dim=1)

        # flatten node features to (B*H*W, C_r)
        xg = x_reduced.reshape(B, -1, H * W).permute(0, 2, 1).reshape(-1, x_reduced.size(1))
        
        # batch assignment per node for pooling
        batch = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, H*W).reshape(-1)

        for conv in self.convs:
            xg = conv(xg, edge_index)
            xg = F.relu(xg)
        # global pooling
        xg = global_mean_pool(xg, batch)                    # [B, C_out]

        # reshape to spatial (B, C_out, 1, 1) and normalize
        xg = xg.view(B, -1, 1, 1)
        xg = self.norm(xg)
        return xg
    
class GCNNetworkV2(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):

        super(GCNNetworkV2, self).__init__()
        self.preprocess = nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True) # conv1x1 to fuse features
        # GCN layers
         # dynamically create exactly C.GCN_layers-1 hidden layers + 1 output layer
        self.convs = nn.ModuleList()
        for _ in range(C.GCN_layers - 1):
            self.convs.append(GCNConv(out_channels//reduction, out_channels//reduction, aggr='mean', bias=True))
        # final maps to full out_channels
        self.convs.append(GCNConv(out_channels//reduction, out_channels, aggr='mean', bias=True))
        
        self.norm = norm_layer(out_channels) 
        # Graph constructor
        self.graph = GraphConstructor(k=0.5, r=1.0)
        # Convolution to fuse mean and max pooled features
        self.fuse = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        """
        Forward pass: x is a batch of spatial features [B, C_in, H, W].
        We build one graph per sample on the H×W grid, run two GCNConv layers, pool, and normalize.
        """
        B, C_in, H, W = x.shape
        # initial conv projection
        x_reduced = self.preprocess(x)                     # [B, C_r, H, W]

        # build batched edge_index for all B graphs
        edge_indices = []
        for b in range(B):
            # per-sample node features: H×W×C_r
            feat = x_reduced[b].permute(1, 2, 0).contiguous()  # [H, W, C_r]
            A = self.graph.create_graph(feat)
            ei, _ = dense_to_sparse(A)
            ei = ei + b * (H * W)
            edge_indices.append(ei)
        edge_index = torch.cat(edge_indices, dim=1)

        # flatten node features to (B*H*W, C_r)
        xg = x_reduced.reshape(B, -1, H * W).permute(0, 2, 1).reshape(-1, x_reduced.size(1))
        
        # batch assignment per node for pooling
        batch = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, H*W).reshape(-1)

        for conv in self.convs:
            xg = conv(xg, edge_index)
            xg = F.relu(xg)
        # global pooling
        x_mean = global_mean_pool(xg, batch)                    # [B, C_out]
        x_max = global_max_pool(xg, batch)                    # [B, C_out]
        # reshape to spatial (B, C_out, 1, 1) and normalize
        x_mean = x_mean.view(B, -1, 1, 1)
        x_max = x_max.view(B, -1, 1, 1)
        # Fuse mean and max pooled features by 1x1 convolution
        x_cat = torch.cat((x_mean, x_max), dim=1)  # (B, 2*C_out, 1,1)
        xg = self.fuse(x_cat)  # (B, C_out, 1,1)
        xg = self.norm(xg)
        
        return xg

class GCNNetworkV3(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):

        super(GCNNetworkV3, self).__init__()
        self.preprocess = nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True) # conv1x1 to fuse features
        # GCN layers
         # dynamically create exactly C.GCN_layers-1 hidden layers + 1 output layer
        self.convs = nn.ModuleList()
        for _ in range(C.GCN_layers - 1):
            self.convs.append(GCNConv(out_channels//reduction, out_channels//reduction, aggr='mean', bias=True))
        # final maps to full out_channels
        self.convs.append(GCNConv(out_channels//reduction, out_channels, aggr='mean', bias=True))
        
        self.norm = norm_layer(out_channels) 
        # Graph constructor
        self.graph = GraphConstructor(k=0.5, r=1.0)
        # Gating mechanism for dynamic fusion of mean and max pooled features
        self.gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels // reduction, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, 2, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass: x is a batch of spatial features [B, C_in, H, W].
        We build one graph per sample on the H×W grid, run two GCNConv layers, pool, and normalize.
        """
        B, C_in, H, W = x.shape
        # initial conv projection
        x_reduced = self.preprocess(x)                     # [B, C_r, H, W]

        # build batched edge_index for all B graphs
        edge_indices = []
        for b in range(B):
            # per-sample node features: H×W×C_r
            feat = x_reduced[b].permute(1, 2, 0).contiguous()  # [H, W, C_r]
            A = self.graph.create_graph(feat)
            ei, _ = dense_to_sparse(A)
            ei = ei + b * (H * W)
            edge_indices.append(ei)
        edge_index = torch.cat(edge_indices, dim=1)

        # flatten node features to (B*H*W, C_r)
        xg = x_reduced.reshape(B, -1, H * W).permute(0, 2, 1).reshape(-1, x_reduced.size(1))
        
        # batch assignment per node for pooling
        batch = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, H*W).reshape(-1)

        for conv in self.convs:
            xg = conv(xg, edge_index)
            xg = F.relu(xg)
        # global pooling
        x_mean = global_mean_pool(xg, batch)                    # [B, C_out]
        x_max = global_max_pool(xg, batch)                    # [B, C_out]

        # reshape to spatial (B, C_out, 1, 1) and normalize
        x_mean = x_mean.view(B, -1, 1, 1)
        x_max = x_max.view(B, -1, 1, 1)

        # Dynamic gating fusion of mean and max pooled features
        x_cat = torch.cat((x_mean, x_max), dim=1)  # (B, 2*C_out, 1,1)
        g = self.gate(x_cat)  # (B, 2, 1,1) # gate of the mean and max pooled features
        g_mean, g_max = g[:, :1, ...], g[:, 1:, ...]
        xg = g_mean * x_mean + g_max * x_max  # (B, C_out, 1,1) # apply the gate to pooled features
        xg = self.norm(xg)
        
        return xg
    
class GCNNetworkV4(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d):

        super(GCNNetworkV4, self).__init__()
        self.preprocess = nn.Conv2d(in_channels, out_channels//reduction, kernel_size=1, bias=True) # conv1x1 to fuse features
        # GCN layers
         # dynamically create exactly C.GCN_layers-1 hidden layers + 1 output layer
        self.convs = nn.ModuleList()
        for _ in range(C.GCN_layers - 1):
            self.convs.append(GCNConv(out_channels//reduction, out_channels//reduction, aggr='mean', bias=True))
        # final maps to full out_channels
        self.convs.append(GCNConv(out_channels//reduction, out_channels, aggr='mean', bias=True))
        
        self.norm = norm_layer(out_channels) 
        # Graph constructor
        self.graph = GraphConstructor(k=0.5, r=1.0)

        # Add SAGPooling module for graph pooling before fusion
        self.sag_pool = SAGPooling(out_channels, ratio=C.sag_pool_ratio)

        # Convolution to fuse mean, max, and SAG pooled features
        self.fuse = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        """
        Forward pass: x is a batch of spatial features [B, C_in, H, W].
        We build one graph per sample on the H×W grid, run two GCNConv layers, pool, and normalize.
        """
        B, C_in, H, W = x.shape
        # initial conv projection
        x_reduced = self.preprocess(x)                     # [B, C_r, H, W]

        # build batched edge_index for all B graphs
        edge_indices = []
        for b in range(B):
            # per-sample node features: H×W×C_r
            feat = x_reduced[b].permute(1, 2, 0).contiguous()  # [H, W, C_r]
            A = self.graph.create_graph(feat)
            ei, _ = dense_to_sparse(A)
            ei = ei + b * (H * W)
            edge_indices.append(ei)
        edge_index = torch.cat(edge_indices, dim=1)

        # flatten node features to (B*H*W, C_r)
        xg = x_reduced.reshape(B, -1, H * W).permute(0, 2, 1).reshape(-1, x_reduced.size(1))
        
        # batch assignment per node for pooling
        batch = torch.arange(B, device=x.device).unsqueeze(1).repeat(1, H*W).reshape(-1)

        for conv in self.convs:
            xg = conv(xg, edge_index)
            xg = F.relu(xg)
        # global pooling
        x_mean = global_mean_pool(xg, batch)                    # [B, C_out]
        x_max = global_max_pool(xg, batch)                    # [B, C_out]
        # Replace direct call to SAGPooling with module usage
        x_sag_nodes, _, _, batch_sag, _, _ = self.sag_pool(xg, edge_index, None, batch)
        x_sag = global_mean_pool(x_sag_nodes, batch_sag)  # [B, C_out]
        # reshape to spatial (B, C_out, 1, 1) and normalize
        x_mean = x_mean.view(B, -1, 1, 1)
        x_max = x_max.view(B, -1, 1, 1)
        x_sag = x_sag.view(B, -1, 1, 1)
        # Fuse mean and max pooled features by 1x1 convolution
        x_cat = torch.cat((x_mean, x_max, x_sag), dim=1)  # (B, 3*C_out, 1,1)
        xg = self.fuse(x_cat)  # (B, C_out, 1,1)
        xg = self.norm(xg)
        
        return xg