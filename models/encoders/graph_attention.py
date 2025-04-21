import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import GATConv, GATv2Conv
from torch_geometric.data import Data, Batch

# Import the ContiguousConv2d class
from ..decoders.MLPDecoder import ContiguousConv2d


class SegformerFeatureExtractor(nn.Module):
    """Extract features from segformer and create a graph."""
    def __init__(self, segformer_model, cfg=None):
        super().__init__()
        self.segformer = segformer_model
        self.cfg = cfg
        # Get the feature level to use for graph creation
        self.feature_level = 3
        if cfg is not None:
            self.feature_level = getattr(cfg, 'graph_feature_level', 3)
        
    def forward(self, rgb, modal_x):
        """
        Extract features from the segformer backbone and create a grid graph.
        
        Args:
            rgb: RGB image tensor [B, C, H, W]
            modal_x: Modal X tensor [B, C, H, W]
            
        Returns:
            features: List of feature maps from segformer at different scales
            rgb_features: RGB node features for graph [B, N, C] (N = H*W)
            x_features: Modal X node features for graph [B, N, C] (N = H*W)
            edge_index: Edge connectivity [2, num_edges]
        """
        # Get multi-scale features from segformer
        features = self.segformer(rgb, modal_x)
        
        # Use the configured feature level (limit to available levels)
        level = min(self.feature_level, len(features) - 1)
        
        # Get the selected level features
        selected_features = features[level]
        B, C, H, W = selected_features.shape
        
        # Create an edge index tensor for the grid graph at this resolution
        edge_index = self._create_grid_graph_connectivity(H, W)
        edge_index = edge_index.to(selected_features.device)
        
        # Try to access individual modality features for the selected level
        try:
            # These represent the pre-fusion features for RGB and X modality
            rgb_features = self.segformer.FRMs[level].rgb_features
            x_features = self.segformer.FRMs[level].x_features
        except (AttributeError, IndexError):
            # Fallback: If direct access isn't possible, use the selected features
            rgb_features = selected_features
            x_features = selected_features
        
        # Reshape to format needed for graph processing
        # [B, C, H, W] -> [B, H*W, C]
        rgb_features = rgb_features.view(B, C, -1).permute(0, 2, 1).contiguous()
        x_features = x_features.view(B, C, -1).permute(0, 2, 1).contiguous()
        
        return features, rgb_features, x_features, edge_index
    
    def _create_grid_graph_connectivity(self, height, width):
        """
        Create a grid graph connectivity matrix based on 8-neighborhood.
        
        Args:
            height: Height of the grid
            width: Width of the grid
            
        Returns:
            edge_index: Edge connectivity [2, num_edges]
        """
        # Total number of nodes
        num_nodes = height * width
        
        # Initialize lists to store edges
        edges = []
        
        # 8-neighborhood connectivity: top, bottom, left, right, and 4 diagonals
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        
        # Create edges based on grid connectivity
        for h in range(height):
            for w in range(width):
                node_idx = h * width + w
                
                for dh, dw in directions:
                    nh, nw = h + dh, w + dw
                    
                    # Check if neighbor is within bounds
                    if 0 <= nh < height and 0 <= nw < width:
                        neighbor_idx = nh * width + nw
                        edges.append([node_idx, neighbor_idx])
        
        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t()
        
        return edge_index


class GATSegmentation(nn.Module):
    """Graph Attention Network for segmentation"""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, heads=4, dropout=0.1, use_gatv2=True):
        super().__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.heads = heads
        self.dropout = dropout
        self.use_gatv2 = use_gatv2
        
        # Use gradient checkpointing to save memory
        self.use_checkpoint = False  # Disabled by default to avoid DDP issues
        
        # Initial projection
        self.in_proj = nn.Linear(in_channels, hidden_channels)
        
        # GATv2 layers
        self.gat_layers = nn.ModuleList()
        
        # First layer: in_channels -> hidden_channels*heads
        if self.use_gatv2:
            self.gat_layers.append(GATv2Conv(
                hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=True
            ))
        else:
            self.gat_layers.append(GATConv(
                hidden_channels, hidden_channels, heads=heads, dropout=dropout, concat=True
            ))
        
        # Middle layers (if any)
        for _ in range(num_layers - 2):
            if self.use_gatv2:
                self.gat_layers.append(GATv2Conv(
                    hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True
                ))
            else:
                self.gat_layers.append(GATConv(
                    hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout, concat=True
                ))
        
        # Last layer: hidden_channels*heads -> out_channels (no concatenation)
        if num_layers > 1:
            if self.use_gatv2:
                self.gat_layers.append(GATv2Conv(
                    hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False
                ))
            else:
                self.gat_layers.append(GATConv(
                    hidden_channels * heads, out_channels, heads=1, dropout=dropout, concat=False
                ))
        else:
            # If only one layer, adjust the output dimensions
            if self.use_gatv2:
                self.gat_layers[0] = GATv2Conv(
                    hidden_channels, out_channels, heads=1, dropout=dropout, concat=False
                )
            else:
                self.gat_layers[0] = GATConv(
                    hidden_channels, out_channels, heads=1, dropout=dropout, concat=False
                )
        
        # Normalization and dropout
        self.norm = nn.LayerNorm(hidden_channels)
        self.dropout = nn.Dropout(dropout)
    
    def _layer_forward(self, x, edge_index, layer_idx):
        """Helper function for gradient checkpointing"""
        x = self.gat_layers[layer_idx](x, edge_index)
        return F.relu(x)
    
    def forward(self, x, edge_index):
        """
        Forward pass through the GAT
        
        Args:
            x: Node features [B, N, C]
            edge_index: Edge connectivity [2, E]
            
        Returns:
            x: Output node features [B, N, out_channels]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Process each batch sample separately since PyG operates on single graphs
        output_list = []
        
        for b in range(batch_size):
            # Get features for this batch [N, C]
            x_b = x[b]
            
            # Initial projection
            x_b = self.in_proj(x_b)
            x_b = self.norm(x_b)
            x_b = F.relu(x_b)
            x_b = self.dropout(x_b)
            
            # Apply GAT layers with gradient checkpointing if enabled
            for i, layer in enumerate(self.gat_layers):
                if self.use_checkpoint and self.training:
                    x_b = checkpoint.checkpoint(self._layer_forward, x_b, edge_index, i)
                else:
                    x_b = layer(x_b, edge_index)
                    x_b = F.relu(x_b)
                
            # Add to output list
            output_list.append(x_b)
        
        # Stack outputs
        output = torch.stack(output_list, dim=0)
        
        return output


class SegformerGAT(nn.Module):
    """
    Segformer-based Graph Attention Network for feature enhancement.
    This module extracts features from the segformer backbone, constructs a graph,
    processes it with a Graph Attention Network (GAT), and returns enhanced features
    to be used by a decoder for final segmentation.
    """
    def __init__(self, segformer_model, in_channels, hidden_channels, out_channels, 
                 num_layers=2, heads=4, dropout=0.1, use_gatv2=True, cfg=None):
        super().__init__()
        
        # Feature extractor from the segformer
        self.feature_extractor = SegformerFeatureExtractor(segformer_model, cfg)
        
        # Get the feature level from config
        self.feature_level = 3
        if cfg is not None:
            self.feature_level = getattr(cfg, 'graph_feature_level', 3)
        
        # Get input channels based on feature level if different from provided
        if self.feature_level != 3:
            # Adjust in_channels based on the selected feature level
            level_channels = [64, 128, 320, 512]  # Default for MiT-B2
            if hasattr(segformer_model, 'channels'):
                level_channels = segformer_model.channels
            level = min(self.feature_level, len(level_channels) - 1)
            in_channels = level_channels[level]
        
        # Store the output channels for upsampling if needed
        self.out_channels = out_channels
        
        # Modality fusion layer for combining RGB and X features
        self.modality_fusion = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),  # Concat RGB and X features
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GAT for processing the graph
        self.gat = GATSegmentation(
            in_channels=hidden_channels,  # After modality fusion
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Match hidden dimensions
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
            use_gatv2=use_gatv2
        )
        
        # Final projection to match output channels
        self.final_proj = ContiguousConv2d(hidden_channels, out_channels, kernel_size=1)
    
    def forward(self, imgs, modal_xs):
        """
        Forward pass through the SegformerGAT
        
        Args:
            imgs: RGB input tensor [B, C, H, W]
            modal_xs: Modal X input tensor [B, C, H, W]
            
        Returns:
            enhanced_features: Enhanced features from GAT processing, in a format
                               compatible with the decoder [B, C, H, W]
        """
        # Get Segformer features and create graph
        features, rgb_features, x_features, edge_index = self.feature_extractor(imgs, modal_xs)
        
        # Get original backbone features for later use (always use the last level for fusion)
        backbone_features = features[-1]  # Last level features
        
        # Combine RGB and X modality features
        # [B, N, C] + [B, N, C] -> [B, N, 2*C]
        combined_features = torch.cat([rgb_features, x_features], dim=2)
        
        # Fuse the modality features
        # [B, N, 2*C] -> [B, N, hidden_C]
        fused_features = self.modality_fusion(combined_features)
        
        # Process with GAT
        gat_output = self.gat(fused_features, edge_index)
        
        # Reshape GAT output back to spatial dimensions - ensure tensor is contiguous
        B = features[self.feature_level].shape[0]
        H, W = features[self.feature_level].shape[2], features[self.feature_level].shape[3]
        gat_output = gat_output.contiguous().view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        
        # Apply final projection
        enhanced_features = self.final_proj(gat_output).contiguous()
        
        # If feature level isn't the last one, we need to upsample or downsample to match the last level size
        if self.feature_level != 3:
            target_h, target_w = features[-1].shape[2], features[-1].shape[3]
            current_h, current_w = enhanced_features.shape[2], enhanced_features.shape[3]
            
            if current_h != target_h or current_w != target_w:
                enhanced_features = F.interpolate(
                    enhanced_features, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ).contiguous()
        
        # Return the enhanced features to be used by the decoder
        return enhanced_features 