import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F

class MLP(nn.Module):
    """
    Linear Embedding: 
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class DecoderHead(nn.Module):
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=512,
                 align_corners=False):
        
        super(DecoderHead, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        # Convolution-based embeddings for spatial preservation
        self.linear_c1 = nn.Conv2d(c1_in_channels, embed_dim, kernel_size=1)
        self.linear_c2 = nn.Conv2d(c2_in_channels, embed_dim, kernel_size=1)
        self.linear_c3 = nn.Conv2d(c3_in_channels, embed_dim, kernel_size=1)
        self.linear_c4 = nn.Conv2d(c4_in_channels, embed_dim, kernel_size=1)
        
        # Dynamic feature fusion
        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.GELU()
        )

        # Auxiliary attention layer
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim, embed_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(embed_dim // 4, embed_dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Prediction head
        self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        
        self.dropout = nn.Dropout2d(dropout_ratio) if dropout_ratio > 0 else None
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        n, _, h, w = c1.size()

        # Feature embedding
        _c1 = self.linear_c1(c1)
        _c2 = F.interpolate(self.linear_c2(c2), size=(h, w), mode='bilinear', align_corners=self.align_corners)
        _c3 = F.interpolate(self.linear_c3(c3), size=(h, w), mode='bilinear', align_corners=self.align_corners)
        _c4 = F.interpolate(self.linear_c4(c4), size=(h, w), mode='bilinear', align_corners=self.align_corners)

        # Fusion with attention
        fused = self.linear_fuse(torch.cat([_c1, _c2, _c3, _c4], dim=1))
        attention = self.attention(fused)
        fused = fused * attention

        if self.dropout is not None:
            fused = self.dropout(fused)
        
        # Final prediction
        x = self.linear_pred(fused)
        return x

        