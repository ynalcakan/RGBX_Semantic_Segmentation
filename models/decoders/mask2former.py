import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Mask2Former(nn.Module):
    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        
        # Add learnable query embeddings
        self.query_embed = nn.Embedding(num_queries, 256)  # 256 is hidden_dim
        nn.init.normal_(self.query_embed.weight, std=0.02)  # Initialize with normal distribution
        
        # Pixel decoder (FPN-style)
        self.pixel_decoder = PixelDecoder(in_channels, norm_layer)
        
        # Transformer decoder
        self.transformer_decoder = TransformerDecoder(
            hidden_dim=256,  # transformer working dimension
            nheads=8,
            num_decoder_layers=9,
            dim_feedforward=2048,
            dropout=0.1
        )
        
        # Prediction heads
        self.class_embed = nn.Linear(256, num_classes + 1)  # +1 for null class
        self.mask_embed = MaskPredictor(256, hidden_dim=256)

    def forward(self, features):
        # Process features through pixel decoder
        mask_features, transformer_features = self.pixel_decoder(features)
        
        # Reshape transformer features for attention
        B, C, H, W = transformer_features.shape
        transformer_features = transformer_features.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        
        # Generate queries from learned embeddings
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        
        # Transformer decoder
        decoder_output = self.transformer_decoder(queries, transformer_features)
        
        # Reshape decoder output for prediction heads
        decoder_output = decoder_output.transpose(0, 1)  # [B, num_queries, C]
        
        # Predict classes and masks
        outputs_class = self.class_embed(decoder_output)  # [B, num_queries, num_classes+1]
        outputs_mask = self.mask_embed(decoder_output, mask_features)  # [B, num_queries, H, W]
        
        # Return both class and mask predictions
        return {
            'pred_logits': outputs_class,  # [B, num_queries, num_classes+1]
            'pred_masks': outputs_mask,    # [B, num_queries, H, W]
        }

    def init_weights(self):
        # Initialize transformer
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        # Initialize prediction heads
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.class_embed.bias, bias_value)

class PixelDecoder(nn.Module):
    def __init__(self, in_channels, norm_layer):
        super().__init__()
        self.in_channels = in_channels  # List of input channels from different scales
        hidden_dim = 256  # Common working dimension
        
        # Lateral convolutions convert each backbone feature to common dimension
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, hidden_dim, 1) 
            for in_ch in in_channels
        ])
        
        # Output convolutions after each fusion
        self.output_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
                norm_layer(hidden_dim),
                nn.ReLU(inplace=True)
            ) for _ in range(len(in_channels)-1)
        ])
        
        # Final layer to generate mask features
        self.mask_features = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            norm_layer(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Layer to generate transformer features
        self.transformer_features = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 1),
            norm_layer(hidden_dim)
        )

    def forward(self, features):
        # features is list of 4 feature maps from backbone, from fine to coarse
        assert len(features) == len(self.in_channels)
        
        # Convert all scales to common dimension
        laterals = [
            lateral_conv(feature)
            for feature, lateral_conv in zip(features, self.lateral_convs)
        ]
        
        # Process from coarse to fine
        for idx in range(len(laterals)-1, 0, -1):
            # Upsample coarser feature
            laterals[idx-1] = laterals[idx-1] + F.interpolate(
                laterals[idx],
                size=laterals[idx-1].shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            # Apply output convolution
            laterals[idx-1] = self.output_convs[idx-1](laterals[idx-1])
        
        # Generate final outputs
        mask_features = self.mask_features(laterals[0])  # For mask prediction
        transformer_features = self.transformer_features(laterals[-1])  # For transformer
        
        return mask_features, transformer_features

class TransformerDecoder(nn.Module):
    def __init__(self, hidden_dim, nheads, num_decoder_layers, dim_feedforward, dropout):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nheads, dim_feedforward, dropout
        )
        self.layers = nn.ModuleList([
            decoder_layer for _ in range(num_decoder_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries, memory):
        # queries: [num_queries, batch_size, hidden_dim]
        # memory: [HW, batch_size, hidden_dim]
        
        output = queries
        for layer in self.layers:
            output = layer(output, memory)
        
        return self.norm(output)

class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dim_feedforward, dropout):
        super().__init__()
        # Self attention
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        
        # Cross attention
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, hidden_dim)
        )
        self.dropout3 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
    def forward(self, queries, memory):
        # Self attention
        q = k = queries
        queries2 = self.self_attn(q, k, value=queries)[0]
        queries = queries + self.dropout1(queries2)
        queries = self.norm1(queries)
        
        # Cross attention
        queries2 = self.cross_attn(
            query=queries,
            key=memory,
            value=memory
        )[0]
        queries = queries + self.dropout2(queries2)
        queries = self.norm2(queries)
        
        # FFN
        queries2 = self.ffn(queries)
        queries = queries + self.dropout3(queries2)
        queries = self.norm3(queries)
        
        return queries

class MaskPredictor(nn.Module):
    def __init__(self, query_dim, hidden_dim):
        super().__init__()
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = nn.Parameter(torch.ones(1) * 20.0)  # learnable temperature
        
    def forward(self, queries, mask_features):
        batch_size = mask_features.shape[0]
        H, W = mask_features.shape[-2:]
        
        # Project queries
        queries = self.query_proj(queries)  # [B, num_queries, hidden_dim]
        queries = self.out_proj(queries)    # [B, num_queries, hidden_dim]
        
        # Scale queries for better mask prediction
        queries = queries * self.scale.sigmoid()
        
        # Generate masks through dot product with scaling
        mask_features = mask_features.flatten(2)  # [B, hidden_dim, H*W]
        masks = torch.bmm(queries, mask_features)  # [B, num_queries, H*W]
        masks = masks.view(batch_size, -1, H, W)  # [B, num_queries, H, W]
        
        return masks