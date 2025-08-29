import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNRelu(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, kernel_size=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TransformerDecoderLayer(nn.Module):
    """
    Lightweight transformer decoder layer with self-attention over queries
    and cross-attention over pixel-level memory.
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, memory: torch.Tensor):
        # query: [B, Q, C], memory: [B, HW, C]
        q = self.norm1(query)
        query2, _ = self.self_attn(q, q, q)
        query = query + self.dropout1(query2)

        q = self.norm2(query)
        query2, _ = self.cross_attn(q, memory, memory)
        query = query + self.dropout2(query2)

        q = self.norm3(query)
        q2 = self.linear2(self.dropout(F.gelu(self.linear1(q))))
        query = query + self.dropout3(q2)
        return query


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers: int, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, memory: torch.Tensor):
        for layer in self.layers:
            query = layer(query, memory)
        return query


class Mask2Former(nn.Module):
    """
    Simplified Mask2Former-style decoder that:
    - builds a pixel-level embedding map via FPN-like fusion
    - decodes a set of learned queries with a transformer decoder
    - predicts per-query class logits and mask logits

    Expected input: list of 4 tensors [c1, c2, c3, c4] from the backbone, where
    c1 has the highest spatial resolution (typically 1/4 of input) and c4 the lowest.
    """
    def __init__(
        self,
        in_channels=[64, 128, 320, 512],
        num_classes: int = 40,
        norm_layer: nn.Module = nn.BatchNorm2d,
        embed_dim: int = 256,
        num_queries: int = 100,
        num_heads: int = 8,
        num_layers: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_queries = num_queries

        c1, c2, c3, c4 = in_channels

        # Lateral projections to a common embedding dimension
        self.lateral4 = ConvBNRelu(c4, embed_dim, norm_layer, kernel_size=1)
        self.lateral3 = ConvBNRelu(c3, embed_dim, norm_layer, kernel_size=1)
        self.lateral2 = ConvBNRelu(c2, embed_dim, norm_layer, kernel_size=1)
        self.lateral1 = ConvBNRelu(c1, embed_dim, norm_layer, kernel_size=1)

        # FPN smoothing convs
        self.smooth3 = ConvBNRelu(embed_dim, embed_dim, norm_layer, kernel_size=3)
        self.smooth2 = ConvBNRelu(embed_dim, embed_dim, norm_layer, kernel_size=3)
        self.smooth1 = ConvBNRelu(embed_dim, embed_dim, norm_layer, kernel_size=3)

        # Query embeddings and transformer decoder
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.decoder = TransformerDecoder(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads, dropout=0.1)

        # Prediction heads
        self.class_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_classes + 1),  # +1 for no-object
        )
        self.mask_embed_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Final projection for pixel embedding map
        self.pixel_proj = ConvBNRelu(embed_dim, embed_dim, norm_layer, kernel_size=1)

    def _build_pixel_embedding(self, c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor) -> torch.Tensor:
        # Laterals
        p4 = self.lateral4(c4)
        p3 = self.lateral3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False)
        p2 = self.lateral2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        p1 = self.lateral1(c1) + F.interpolate(p2, size=c1.shape[-2:], mode='bilinear', align_corners=False)

        # Smooth
        p3 = self.smooth3(p3)
        p2 = self.smooth2(p2)
        p1 = self.smooth1(p1)

        # Use highest resolution feature as pixel embedding
        pixel_embed = self.pixel_proj(p1)
        return pixel_embed

    def forward(self, inputs):
        # inputs: [c1, c2, c3, c4]
        c1, c2, c3, c4 = inputs

        # Build pixel-level memory
        pixel_embed = self._build_pixel_embedding(c1, c2, c3, c4)  # [B, C, H, W]
        B, C, H, W = pixel_embed.shape

        # Flatten memory for cross-attention
        memory = pixel_embed.flatten(2).transpose(1, 2)  # [B, HW, C]

        # Prepare queries
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # [B, Q, C]

        # Decode
        decoded_queries = self.decoder(queries, memory)  # [B, Q, C]

        # Class logits per query
        pred_logits = self.class_head(decoded_queries)  # [B, Q, num_classes+1]

        # Mask embeddings per query
        mask_embed = self.mask_embed_head(decoded_queries)  # [B, Q, C]

        # Produce per-query mask logits via dot-product with pixel embeddings
        pixel_flat = pixel_embed.view(B, C, H * W)  # [B, C, HW]
        pred_masks = torch.bmm(mask_embed, pixel_flat)  # [B, Q, HW]
        pred_masks = pred_masks.view(B, self.num_queries, H, W)

        return {
            'pred_logits': pred_logits,
            'pred_masks': pred_masks,
        }


