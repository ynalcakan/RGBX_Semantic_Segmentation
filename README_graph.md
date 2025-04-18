# Graph-Enhanced RGB-X Semantic Segmentation

This module introduces a graph-based approach to enhance multi-modal semantic segmentation. The implementation combines the power of Transformer-based backbones (Segformer) with Graph Attention Networks (GAT) to better capture relationships between RGB and X modality features.

## Architecture Overview

The architecture consists of the following main components and data flow:

1. **Feature Extraction**: The Segformer backbone extracts multi-scale features from both RGB and X modalities.
   - Produces features at multiple scales (typically 4 levels)
   - Captures hierarchical visual patterns through a CNN-Transformer hybrid approach

2. **Graph Construction & Early Fusion**: A grid graph is created where:
   - Each node corresponds to a spatial location
   - RGB and X modality features are concatenated at each node (early fusion)
   - 8-neighborhood connectivity establishes spatial relationships

3. **Graph Attention Processing**: The combined node features are processed using GAT/GATv2:
   - Message passing between spatial locations via graph edges
   - Attention mechanism learns to focus on relevant relationships
   - Enhances feature representation with spatial and cross-modal context

4. **Late Fusion (Feature Integration)**: The graph-enhanced features are integrated with the backbone features:
   - Only the last level of backbone features is modified (x[-1])
   - Fusion mode (weighted/add/concat) determines how backbone and graph features are combined
   - The result maintains multi-scale information while incorporating graph-based enhancements

5. **Segmentation Decoding**: The enhanced multi-scale features are processed by a decoder:
   - Receives the full set of backbone features (with last level enhanced)
   - Produces the final segmentation output utilizing both backbone and graph knowledge

## Detailed Data Flow

```
RGB Image ─┬─> Segformer Backbone ─┬─> x[0],x[1],x[2],x[3] ─┐
           │                       │                        │
X Modality ─┘                      └─> FRM ─> RGB & X ──┐   │
                                           Features      │   │
                                                         ▼   │
                                                       Graph │
                                                    Creation │
                                                         │   │
                                                         ▼   │
                                                   Early Fusion
                                                   (Node Level)
                                                         │
                                                         ▼
                                                  GAT Processing
                                                         │
                                                         ▼
                                                 graph_features
                                                         │
                                                         ▼
                                                   Late Fusion ◄─── x[-1]
                                              (Feature Map Level)
                                                         │
                                                         ▼
                                            Enhanced x[-1] → x[0:3]
                                                         │
                                                         ▼
                                                      Decoder
                                                         │
                                                         ▼
                                                  Segmentation
```

## Fusion Stages Explained

The architecture employs two distinct fusion stages that serve different purposes:

### 1. Early Fusion (Input Modality Fusion)

**What**: Combines RGB and X modality features at each node of the graph

**Where**: Before GAT processing, inside the graph structure

**Implementation**:
```python
# Inside SegformerGAT forward method
combined_features = torch.cat([rgb_features, x_features], dim=2)  # [B, N, 2*C]
fused_features = self.modality_fusion(combined_features)  # [B, N, C]
```

**Characteristics**:
- Always uses concatenation followed by projection
- Fixed part of the architecture (not configurable)
- Enables the GAT to process relationships between both modalities
- Operates at the node level (per spatial location)

### 2. Late Fusion (Backbone-Graph Feature Fusion)

**What**: Combines backbone features with graph-processed features

**Where**: After both backbone and GAT processing, before decoder

**Implementation**:
```python
# Inside builder.py encode_decode method
if self.fusion_mode == 'weighted':
    x[-1] = (self.fusion_weight * x[-1] + (1 - self.fusion_weight) * graph_features).contiguous()
elif self.fusion_mode == 'add':
    x[-1] = (x[-1] + graph_features).contiguous()
```

**Characteristics**:
- Configurable via `C.graph_fusion_mode` setting
- Only modifies the last level of backbone features (x[-1])
- Preserves multi-scale information while incorporating graph knowledge
- Operates at the feature map level (entire spatial extent)

## Key Components

### SegformerFeatureExtractor

Extracts features from the segformer backbone and creates a grid graph structure:

- Processes RGB and X modality inputs through the segformer backbone
- Accesses individual RGB and X features before fusion from the FRM (Feature Rectify Module)
- Creates a grid graph connectivity matrix (8-neighborhood)
- Provides features and graph structure for GAT processing

```python
# Core feature extraction and graph creation
features, rgb_features, x_features, edge_index = self.feature_extractor(imgs, modal_xs)
```

### GATSegmentation

Processes the graph features using GAT/GATv2:

- Takes node features and edge connectivity as input
- Applies multiple GAT layers with configurable heads and dimensions
- Handles batched graph data processing efficiently
- Returns enhanced node features that incorporate spatial and cross-modal context

```python
# Process with GAT
gat_output = self.gat(fused_features, edge_index)
```

### SegformerGAT

Orchestrates the entire graph processing pipeline:

- Extracts features from the segformer backbone
- Performs early fusion of RGB and X modality features
- Processes the graph using GAT
- Projects the graph output back to the feature space compatible with the decoder
- Ensures tensors remain contiguous throughout processing for optimal memory layout

```python
# Main forward pass
enhanced_features = self.graph_processor(rgb, modal_x)
```

## Late Fusion Methods

The architecture supports several methods for integrating graph-enhanced features with backbone features (late fusion):

### 1. Weighted Fusion

Uses a learnable parameter to balance backbone and graph features:

```python
x[-1] = (self.fusion_weight * x[-1] + (1 - self.fusion_weight) * graph_features).contiguous()
```

**Characteristics**:
- Employs a single learnable parameter (`fusion_weight`) optimized during training
- The parameter determines the relative importance of backbone vs. graph features
- Maintains the original feature dimensions (no dimension change)
- Automatically adapts to the optimal balance between traditional CNN features and graph-enhanced features
- The `fusion_weight` value after training provides insight into the relative importance of each path

### 2. Additive Fusion

Simply adds backbone and graph features:

```python
x[-1] = (x[-1] + graph_features).contiguous()
```

**Characteristics**:
- No additional parameters
- Equal contribution from both feature streams
- Simple and computationally efficient
- Works well when both feature streams have complementary information

### 3. Concatenation Fusion

Combines features by concatenating and projecting (requires additional implementation):

```python
# Conceptual example
combined = torch.cat([x[-1], graph_features], dim=1)
x[-1] = self.fusion_conv(combined)
```

**Characteristics**:
- Preserves all information from both feature streams before projection
- Requires additional convolution layers to process the combined features
- More expressive but adds parameters and computation
- Allows for more complex feature integration patterns

## Configuration Options

The graph-enhanced segmentation approach can be configured through the following parameters in `config.py`:

```python
# Graph segmentation configs
C.create_graph = True  # Enable graph creation
C.feature_dim = 512  # Node feature dimension
C.gat_hidden_dim = 512  # GAT hidden dimension
C.gat_num_layers = 2  # Number of GAT layers
C.gat_heads = 4  # Number of attention heads per layer
C.gat_dropout = 0.1  # Dropout rate for GAT
C.use_gatv2 = True  # Use GATv2 instead of GAT
C.graph_fusion_mode = 'weighted'  # Options: 'add', 'weighted', 'concat'
```

The `C.graph_fusion_mode` setting specifically controls the **late fusion** stage (not the early fusion stage), determining how backbone and graph-enhanced features are combined before the decoder.

## Training and Inference

The graph-enhanced model is fully integrated with the existing training and evaluation pipelines. No special handling is required; the model automatically creates and processes the graph during both training and inference.

### Training

Standard training procedure with encoder-decoder architecture:
```python
# Process with graph-based enhancement if enabled
if self.use_graph:
    # Get enhanced features from graph processor
    graph_features = self.graph_processor(rgb, modal_x)
    # Enhance the backbone features based on fusion mode
    if self.fusion_mode == 'weighted':
        x[-1] = (self.fusion_weight * x[-1] + (1 - self.fusion_weight) * graph_features).contiguous()
```

The model uses PyTorch's Distributed Data Parallel (DDP) for efficient multi-GPU training with `find_unused_parameters=False` for optimal performance.

### Inference

The inference process uses the same forward pass as training:
```python
# The same forward pass is used for both training and inference
out = model(imgs, modal_xs)
```

## Implementation Details

### Feature Access

The model accesses pre-fusion features for both modalities from the Feature Rectify Module (FRM):

```python
# Store the original features for later access by graph module
self.rgb_features = None
self.x_features = None

def forward(self, x1, x2):
    # Store the input features for later access
    self.rgb_features = x1.detach()
    self.x_features = x2.detach()
```

This approach allows the graph module to access features before they are fused in the standard backbone path.

### Graph Connectivity

An 8-neighborhood grid graph is constructed to represent spatial relationships:

```python
# 8-neighborhood connectivity: top, bottom, left, right, and 4 diagonals
directions = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]
```

This connectivity pattern allows message passing between adjacent pixels in all directions, providing rich spatial context.

### Early Modality Fusion

RGB and X modality features are combined at each node:

```python
# Combine RGB and X modality features
# [B, N, C] + [B, N, C] -> [B, N, 2*C]
combined_features = torch.cat([rgb_features, x_features], dim=2)

# Fuse the modality features
# [B, N, 2*C] -> [B, N, C]
fused_features = self.modality_fusion(combined_features)
```

This early fusion enables the GAT to learn relationships between modalities directly.

### Memory Optimization

To ensure optimal memory layout for distributed training and eliminate stride mismatch warnings, tensors are made contiguous at key operations:

```python
# Reshape GAT output back to spatial dimensions - ensure tensor is contiguous
gat_output = gat_output.contiguous().view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

# Apply final projection to match feature dimensions if needed
enhanced_features = self.final_proj(gat_output).contiguous()
```

## Requirements

- PyTorch (>=1.7.0)
- PyTorch Geometric (for GAT/GATv2 implementation)
- timm (for Segformer backbone)

## Advantages

1. **Explicit Cross-Modal Interaction**: By combining RGB and X modality features in the graph nodes, the GAT can learn relationships between modalities directly.

2. **Spatial Context**: The graph structure captures spatial relationships between regions, enabling better boundary delineation.

3. **Adaptive Feature Integration**: The weighted fusion mechanism learns the optimal balance between backbone and graph-enhanced features.

4. **Multi-Scale Preservation**: Only enhancing the last level of backbone features preserves the multi-scale information while incorporating graph knowledge.

5. **Flexible Architecture**: The approach can be used with different backbones, decoders, and modalities with minimal changes.

6. **Integration with Existing Pipeline**: The graph-enhanced approach is fully integrated with the existing segmentation pipeline.

7. **Distributed Training Optimized**: The implementation is optimized for distributed training with contiguous tensors and efficient DDP configuration. 