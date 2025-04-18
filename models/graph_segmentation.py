import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing


class GraphSegmentationNet(nn.Module):
    def __init__(self, in_channels=32, hidden_channels=64, out_channels=9, num_layers=3, 
                 dropout=0.1, conv_type='sage'):
        super(GraphSegmentationNet, self).__init__()
        
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection layer
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        
        # Choose graph convolution type
        if conv_type == 'gcn':
            conv_layer = GCNConv
        elif conv_type == 'gat':
            conv_layer = GATConv
        elif conv_type == 'sage':
            conv_layer = SAGEConv
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
        
        # First conv layer (from input dim to hidden dim)
        self.convs.append(conv_layer(hidden_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(conv_layer(hidden_channels, hidden_channels))
        
        # Final conv layer for classification
        self.convs.append(conv_layer(hidden_channels, out_channels))
        
        # Batch normalization layers
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers - 1)
        ])
        
    def forward(self, graph_data_list):
        """
        Forward pass for a batch of graph data.
        
        Args:
            graph_data_list: List of torch_geometric.data.Data objects
            
        Returns:
            A tensor of shape [batch_size, height, width, num_classes] with class logits
        """
        # Convert list of graphs to batched graph
        if not isinstance(graph_data_list, Batch):
            batch_data = Batch.from_data_list(graph_data_list)
        else:
            batch_data = graph_data_list
        
        # Extract node features and edge indices
        x, edge_index = batch_data.x, batch_data.edge_index
        
        # Initial projection
        x = self.input_proj(x)
        
        # Apply graph convolutions
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final convolution for class prediction
        x = self.convs[-1](x, edge_index)
        
        # The output is now node-level predictions
        # We need to reshape it to the original image shape
        
        # Get batch assignment for each node
        batch_indices = batch_data.batch
        
        # Get grid sizes for each graph
        grid_sizes = batch_data.grid_size
        
        # Reshape predictions back to grid shape for each graph in the batch
        batch_size = batch_indices.max().item() + 1
        outputs = []
        
        for batch_idx in range(batch_size):
            # Get node predictions for this graph
            batch_mask = batch_indices == batch_idx
            graph_preds = x[batch_mask]
            
            # Get grid size for this graph
            height, width = grid_sizes[batch_idx]
            
            # Reshape to 2D grid (height, width, num_classes)
            grid_preds = graph_preds.reshape(height, width, self.out_channels)
            outputs.append(grid_preds)
        
        # Stack outputs for batch
        return torch.stack(outputs)


class GraphAwareSegmentation(nn.Module):
    """
    A module that combines traditional CNN-based segmentation with graph-based segmentation.
    """
    def __init__(self, cnn_model, graph_model, fusion_mode='add'):
        super(GraphAwareSegmentation, self).__init__()
        self.cnn_model = cnn_model
        self.graph_model = graph_model
        self.fusion_mode = fusion_mode
        
        # Option 1: Simple addition of logits
        if fusion_mode == 'add':
            self.fusion = lambda x, y: x + y
            
        # Option 2: Learnable weighted fusion
        elif fusion_mode == 'weighted':
            self.alpha = nn.Parameter(torch.tensor(0.5))
            self.fusion = lambda x, y: self.alpha * x + (1 - self.alpha) * y
            
        # Option 3: Channel-wise attention fusion
        elif fusion_mode == 'attention':
            num_classes = graph_model.out_channels
            self.attention = nn.Sequential(
                nn.Conv2d(num_classes * 2, num_classes, kernel_size=3, padding=1),
                nn.BatchNorm2d(num_classes),
                nn.ReLU(),
                nn.Conv2d(num_classes, num_classes, kernel_size=1),
                nn.Sigmoid()
            )
            self.fusion = self._attention_fusion
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    def _attention_fusion(self, cnn_out, graph_out):
        # Concatenate CNN and graph outputs along channel dimension
        concat = torch.cat([cnn_out, graph_out], dim=1)
        # Generate attention weights
        weights = self.attention(concat)
        # Apply attention
        return weights * cnn_out + (1 - weights) * graph_out
    
    def forward(self, rgb_x, x_modal, graph_data=None):
        # Get CNN segmentation output
        cnn_out = self.cnn_model(rgb_x, x_modal)
        
        # If graph data is not available, return only CNN output
        if graph_data is None:
            return cnn_out
        
        # Get graph segmentation output
        graph_out = self.graph_model(graph_data)
        
        # Resize graph output to match CNN output if necessary
        if graph_out.shape[1:3] != cnn_out.shape[2:4]:
            graph_out = F.interpolate(
                graph_out.permute(0, 3, 1, 2),  # [B, H, W, C] -> [B, C, H, W]
                size=cnn_out.shape[2:4],
                mode='bilinear',
                align_corners=False
            )
        else:
            # Permute graph_out from [B, H, W, C] to [B, C, H, W]
            graph_out = graph_out.permute(0, 3, 1, 2)
        
        # Fuse CNN and graph outputs
        fused_out = self.fusion(cnn_out, graph_out)
        
        return fused_out 