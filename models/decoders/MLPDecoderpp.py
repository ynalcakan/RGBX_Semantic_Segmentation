import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class AdaptiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(4))
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(dim * 4, dim, 1, bias=False),
            nn.LayerNorm([dim, 1, 1]),
            nn.GELU()
        )
        
    def forward(self, features):
        weights = F.softmax(self.weights, dim=0)
        weighted_features = [w * f for w, f in zip(weights, features)]
        fused = torch.cat(weighted_features, dim=1)
        return self.fusion_conv(fused)

class MLP(nn.Module):
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        self.attention = SelfAttention(embed_dim)
        
    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x + self.attention(x)  # Residual connection
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        dilations = [1, 6, 12, 18]
        
        self.aspp = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
                nn.LayerNorm([out_channels, 1, 1]),
                nn.GELU()
            ) for dilation in dilations
        ])
        
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.LayerNorm([out_channels, 1, 1]),
            nn.GELU()
        )
        
        self.output_conv = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.LayerNorm([out_channels, 1, 1]),
            nn.GELU()
        )
        
    def forward(self, x):
        size = x.size()[2:]
        
        res = []
        for aspp_module in self.aspp:
            res.append(aspp_module(x))
        
        res.append(F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=True))
        
        return self.output_conv(torch.cat(res, dim=1))

class ProgressiveFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fusion_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(dim * 2, dim, 3, padding=1),
                nn.LayerNorm([dim, 1, 1]),
                nn.GELU()
            ) for _ in range(3)
        ])
        
    def forward(self, features):
        # Progressive fusion from deep to shallow
        current = features[-1]
        for i, (feature, fusion) in enumerate(zip(reversed(features[:-1]), self.fusion_layers)):
            current = fusion(torch.cat([current, feature], dim=1))
        return current

class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        
        # Add ASPP module for better multi-scale feature extraction
        self.aspp = ASPP(embedding_dim, embedding_dim)
        
        # Progressive fusion pathway
        self.progressive_fusion = ProgressiveFusion(embedding_dim)
        
        # Adaptive fusion for final combination
        self.adaptive_fusion = AdaptiveFusion(embedding_dim)
        
        # Auxiliary heads for deep supervision
        self.aux_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
                nn.LayerNorm([embedding_dim // 2, 1, 1]),
                nn.GELU(),
                nn.Conv2d(embedding_dim // 2, num_classes, 1)
            ) for _ in range(3)
        ])
        
        self.linear_pred = nn.Sequential(
            nn.Conv2d(embedding_dim, embedding_dim // 2, 3, padding=1),
            nn.LayerNorm([embedding_dim // 2, 1, 1]),
            nn.GELU(),
            nn.Conv2d(embedding_dim // 2, self.num_classes, 1)
        )
       
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        # Process features through MLPs
        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        # Apply ASPP to deepest features
        _c4 = self.aspp(_c4)
        
        # Interpolate features to the same size
        _c4_up = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        _c3_up = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        _c2_up = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        
        # Progressive fusion pathway
        prog_features = self.progressive_fusion([_c1, _c2_up, _c3_up, _c4_up])
        
        # Adaptive fusion pathway
        adapt_features = self.adaptive_fusion([_c4_up, _c3_up, _c2_up, _c1])
        
        # Combine both pathways
        _c = prog_features + adapt_features
        
        if self.training:
            # Generate auxiliary outputs for deep supervision
            aux_outputs = []
            aux_features = [_c2_up, _c3_up, _c4_up]
            for feat, aux_head in zip(aux_features, self.aux_heads):
                aux_outputs.append(aux_head(feat))
        
        if self.dropout is not None:
            _c = self.dropout(_c)
            
        x = self.linear_pred(_c)

        if self.training:
            return x, aux_outputs
        return x

        