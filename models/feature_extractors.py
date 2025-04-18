import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

class SimpleCNN(nn.Module):
    """
    A simple CNN-based feature extractor (default implementation).
    """
    def __init__(self, in_channels=3, feature_dim=32):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, feature_dim, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(feature_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1).squeeze(-1)
        return x


class ResNetExtractor(nn.Module):
    """
    A ResNet-based feature extractor.
    Uses a truncated ResNet model to extract features.
    """
    def __init__(self, in_channels=3, feature_dim=32, pretrained=True, freeze_backbone=False):
        super(ResNetExtractor, self).__init__()
        # Start with a pretrained ResNet-18
        resnet = models.resnet18(pretrained=pretrained)
        
        # Use the stem and first few layers of ResNet
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            resnet.layer1
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Calculate the output dimensions after the backbone
        # For small patches, we need to make sure the spatial dims don't become too small
        # ResNet reduces spatial dims by factor of 4
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer to get the desired feature dimension
        backbone_feat_dim = 64  # Output of ResNet's layer1
        self.projection = nn.Linear(backbone_feat_dim, feature_dim)
        
    def forward(self, x):
        # Check input size to avoid dimension errors with small patches
        batch_size, channels, height, width = x.shape
        if height < 32 or width < 32:
            # For very small patches, upsample to avoid dimension issues
            x = F.interpolate(x, size=(max(32, height), max(32, width)), mode='bilinear', align_corners=False)
        
        # Extract features through the backbone
        x = self.backbone(x)
        x = self.pool(x).view(batch_size, -1)
        x = self.projection(x)
        return x


class MobileNetExtractor(nn.Module):
    """
    A MobileNetV2-based feature extractor.
    Lighter weight than ResNet, suitable for faster processing.
    """
    def __init__(self, in_channels=3, feature_dim=32, pretrained=True, freeze_backbone=False):
        super(MobileNetExtractor, self).__init__()
        # Start with a pretrained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # Use only the features part (before classifier)
        self.backbone = nn.Sequential(
            # First layer to handle custom number of input channels
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            # First inverted residual block
            mobilenet.features[1]
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Global pooling and projection
        self.pool = nn.AdaptiveAvgPool2d(1)
        backbone_feat_dim = 16  # Output of the first inverted residual block
        self.projection = nn.Linear(backbone_feat_dim, feature_dim)
        
    def forward(self, x):
        # Check input size
        batch_size, channels, height, width = x.shape
        if height < 32 or width < 32:
            x = F.interpolate(x, size=(max(32, height), max(32, width)), mode='bilinear', align_corners=False)
        
        x = self.backbone(x)
        x = self.pool(x).view(batch_size, -1)
        x = self.projection(x)
        return x


class SimpleViT(nn.Module):
    """
    A simple Vision Transformer (ViT) implementation for small patches.
    """
    def __init__(self, in_channels=3, feature_dim=32, patch_size=4, dim=64, depth=2, heads=4):
        super(SimpleViT, self).__init__()
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding will be learned
        self.pos_embedding = nn.Parameter(torch.randn(1, 16, dim))  # Assuming max 4x4 patches
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4, 
                                                 dropout=0.1, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final projection to feature dimension
        self.projection = nn.Linear(dim, feature_dim)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # Create patch embeddings
        x = self.patch_embed(x)  # B, dim, h', w'
        
        # Reshape to sequence format
        h, w = x.shape[2], x.shape[3]
        x = x.permute(0, 2, 3, 1)  # B, h', w', dim
        x = x.reshape(batch_size, h * w, -1)  # B, h'*w', dim
        
        # Add positional embeddings - take only what we need
        pos_embed = self.pos_embedding[:, :h*w, :]
        x = x + pos_embed
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global pooling (mean of all token embeddings)
        x = x.mean(dim=1)
        
        # Project to desired feature dimension
        x = self.projection(x)
        
        return x


def get_feature_extractor(extractor_type, in_channels=3, feature_dim=32, 
                          pretrained=True, freeze_backbone=False):
    """
    Factory function to create the specified feature extractor.
    
    Args:
        extractor_type: Type of feature extractor ('SimpleCNN', 'ResNet', 'MobileNet', 'ViT')
        in_channels: Number of input channels
        feature_dim: Output feature dimension
        pretrained: Whether to use pretrained weights (for supported models)
        freeze_backbone: Whether to freeze backbone weights
        
    Returns:
        A feature extractor model
    """
    if extractor_type == 'SimpleCNN':
        return SimpleCNN(in_channels=in_channels, feature_dim=feature_dim)
    elif extractor_type == 'ResNet':
        return ResNetExtractor(in_channels=in_channels, feature_dim=feature_dim, 
                              pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif extractor_type == 'MobileNet':
        return MobileNetExtractor(in_channels=in_channels, feature_dim=feature_dim, 
                                 pretrained=pretrained, freeze_backbone=freeze_backbone)
    elif extractor_type == 'ViT':
        return SimpleViT(in_channels=in_channels, feature_dim=feature_dim)
    else:
        raise ValueError(f"Unknown feature extractor type: {extractor_type}") 