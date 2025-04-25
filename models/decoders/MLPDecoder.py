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

class SimpleHead(nn.Module):
    """
    Simple classification head
    """
    def __init__(self, in_channels, embed_dim, norm_layer):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.head(x)

class DecoderHead(nn.Module):
    """
    Original MLP Decoder: All stages → MLP → concat → MLP (Design A)
    """
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
        
        self.linear_fuse = nn.Sequential(
                            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1),
                            norm_layer(embedding_dim),
                            nn.ReLU(inplace=True)
                            )
                            
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:],mode='bilinear',align_corners=self.align_corners)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)

        return x

class DecoderHeadB(nn.Module):
    """
    Decoder B: Stage 4 → head, Stage 3 → MLP, both → MLP
    """
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHeadB, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        
        # Head for stage 4
        self.head_c4 = SimpleHead(c4_in_channels, embed_dim, norm_layer)
        
        # MLP for stage 3
        self.mlp_c3 = MLP(input_dim=c3_in_channels, embed_dim=embed_dim)
        
        # Final MLP for combining stage 3 and 4
        self.final_mlp = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
                            
        self.classifier = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        
        # Force all inputs to be used in the computation graph
        # This is just to satisfy DDP's requirement
        dummy_sum = 0
        for c in [c1, c2]:
            dummy_sum = dummy_sum + c.sum() * 0.0
        
        # Extract shapes
        n, _, h, w = c3.shape
        
        # Process stage 4 through head
        _c4 = self.head_c4(c4)
        _c4 = F.interpolate(_c4, size=c3.size()[2:], mode='bilinear', align_corners=self.align_corners)
        
        # Process stage 3 through MLP
        _c3 = self.mlp_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        
        # Combine features
        combined = torch.cat([_c3, _c4], dim=1)
        
        # Apply final MLP
        x = self.final_mlp(combined)
        
        # Upsample to original input resolution (assuming c1 has the highest resolution)
        x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Final classification
        x = self.classifier(x)

        # Add the dummy sum to the output (it's zero so doesn't affect results)
        return x + dummy_sum

class DecoderHeadC(nn.Module):
    """
    Decoder C: Stage 4 → concat, Stage 3 → head, both → MLP
    """
    def __init__(self,
                 in_channels=[64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 norm_layer=nn.BatchNorm2d,
                 embed_dim=768,
                 align_corners=False):
        
        super(DecoderHeadC, self).__init__()
        self.num_classes = num_classes
        self.dropout_ratio = dropout_ratio
        self.align_corners = align_corners
        self.in_channels = in_channels
        
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels
        
        # Concat preparation for stage 4
        self.concat_prep = nn.Sequential(
            nn.Conv2d(c4_in_channels, embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
        
        # Head for stage 3
        self.head_c3 = SimpleHead(c3_in_channels, embed_dim, norm_layer)
        
        # Final MLP after combining stage 3 and 4
        self.final_mlp = nn.Sequential(
            nn.Conv2d(in_channels=embed_dim*2, out_channels=embed_dim, kernel_size=1),
            norm_layer(embed_dim),
            nn.ReLU(inplace=True)
        )
                            
        self.classifier = nn.Conv2d(embed_dim, self.num_classes, kernel_size=1)
       
    def forward(self, inputs):
        # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = inputs
        
        # Force all inputs to be used in the computation graph
        # This is just to satisfy DDP's requirement
        dummy_sum = 0
        for c in [c1, c2]:
            dummy_sum = dummy_sum + c.sum() * 0.0
        
        # Process stage 4 for concatenation
        _c4 = self.concat_prep(c4)
        _c4 = F.interpolate(_c4, size=c3.size()[2:], mode='bilinear', align_corners=self.align_corners)
        
        # Process stage 3 through head
        _c3 = self.head_c3(c3)
        
        # Combine features
        combined = torch.cat([_c3, _c4], dim=1)
        
        # Apply final MLP
        x = self.final_mlp(combined)
        
        # Upsample to original input resolution (assuming c1 has the highest resolution)
        x = F.interpolate(x, size=c1.size()[2:], mode='bilinear', align_corners=self.align_corners)
        
        if self.dropout is not None:
            x = self.dropout(x)
            
        # Final classification
        x = self.classifier(x)

        # Add the dummy sum to the output (it's zero so doesn't affect results)
        return x + dummy_sum

        