import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
from utils.loss_opr import Mask2FormerLoss

from engine.logger import get_logger
from .encoders.graph_attention import SegformerGAT

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.cfg = cfg
        self.use_graph = getattr(cfg, 'create_graph', False)
        
        # import backbone and decoder
        if cfg.backbone == 'swin_s':
            logger.info('Using backbone: Swin-Transformer-small')
            from .encoders.dual_swin import swin_s as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'swin_b':
            logger.info('Using backbone: Swin-Transformer-Base')
            from .encoders.dual_swin import swin_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'segnext_tiny':
            logger.info('Using backbone: SegNeXt-Tiny')
            from .encoders.dual_segnext import segnext_tiny as backbone
            self.channels = [32, 64, 160, 256]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'segnext_s':
            logger.info('Using backbone: SegNeXt-Small')
            from .encoders.dual_segnext import segnext_s as backbone
            self.channels = [64, 128, 320, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'segnext_b':
            logger.info('Using backbone: SegNeXt-Base')
            from .encoders.dual_segnext import segnext_b as backbone
            self.channels = [128, 256, 512, 1024]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'segnext_large':
            logger.info('Using backbone: SegNeXt-Large')
            from .encoders.dual_segnext import segnext_large as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'resnet50':
            logger.info('Using backbone: ResNet-50')
            from .encoders.dual_resnet import dual_resnet50 as backbone
            self.channels = [256, 512, 1024, 2048]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'resnet101':
            logger.info('Using backbone: ResNet-101')
            from .encoders.dual_resnet import dual_resnet101 as backbone
            self.channels = [256, 512, 1024, 2048]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'resnet152':
            logger.info('Using backbone: ResNet-152')
            from .encoders.dual_resnet import dual_resnet152 as backbone
            self.channels = [256, 512, 1024, 2048]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b5':
            logger.info('Using backbone: Segformer-B5')
            from .encoders.dual_segformer import mit_b5 as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.channels = [96, 192, 384, 768]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b3':
            logger.info('Using backbone: Segformer-B3')
            from .encoders.dual_segformer import mit_b3 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b0':
            logger.info('Using backbone: Segformer-B0')
            self.channels = [32, 64, 160, 256]
            from .encoders.dual_segformer import mit_b0 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)
        else:
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.backbone = backbone(norm_fuse=norm_layer)

        self.aux_head = None

        # Get decoder embed dim from config, defaulting to feature_dim if available
        decoder_embed_dim = getattr(cfg, 'feature_dim', getattr(cfg, 'decoder_embed_dim', 512))
        
        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=decoder_embed_dim)
        elif cfg.decoder == 'MLPDecoderpp':
            logger.info('Using MLP Decoderpp')
            from .decoders.MLPDecoderpp import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=decoder_embed_dim)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)
        
        elif cfg.decoder == 'deeplabv3+':
            logger.info('Using Decoder: DeepLabV3+')
            from .decoders.deeplabv3plus import DeepLabV3Plus as Head
            self.decode_head = Head(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)
            from .decoders.fcnhead import FCNHead
            self.aux_index = 2
            self.aux_rate = 0.4
            self.aux_head = FCNHead(self.channels[2], cfg.num_classes, norm_layer=norm_layer)

        elif cfg.decoder == 'mask2former':
            logger.info('Using Mask2Former Decoder')
            from .decoders.mask2former import Mask2Former
            self.decode_head = Mask2Former(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer)

        else:
            logger.info('No decoder(FCN-32s)')
            from .decoders.fcnhead import FCNHead
            self.decode_head = FCNHead(in_channels=self.channels[-1], kernel_size=3, num_classes=cfg.num_classes, norm_layer=norm_layer)

        # Initialize SegformerGAT if graph-based processing is enabled
        if self.use_graph:
            logger.info('Initializing SegformerGAT for graph-based segmentation')
            
            # Graph processing features
            in_channels = self.channels[-1]  # Use the last level channel dimension from backbone
            hidden_channels = getattr(cfg, 'gat_hidden_dim', decoder_embed_dim)  # Match feature dimension
            out_channels = in_channels  # Output channels match input channels for fusion
            num_layers = getattr(cfg, 'gat_num_layers', 2)
            heads = getattr(cfg, 'gat_heads', 4)
            dropout = getattr(cfg, 'gat_dropout', 0.1)
            use_gatv2 = getattr(cfg, 'use_gatv2', True)
            
            # Create SegformerGAT module
            self.graph_processor = SegformerGAT(
                segformer_model=self.backbone,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                num_layers=num_layers,
                heads=heads,
                dropout=dropout,
                use_gatv2=use_gatv2,
                cfg=cfg
            )
            
            # Fusion mode
            self.fusion_mode = getattr(cfg, 'graph_fusion_mode', 'add')
            
            # For weighted fusion, initialize weight
            if self.fusion_mode == 'weighted':
                self.fusion_weight = nn.Parameter(torch.tensor(0.5))

        if cfg.decoder == 'mask2former':
            self.criterion = Mask2FormerLoss(num_classes=cfg.num_classes, ignore_index=255)
        else:
            self.criterion = criterion
        
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)
    
    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info('Loading pretrained model: {}'.format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
        logger.info('Initing weights ...')
        init_weight(self.decode_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.aux_head:
            init_weight(self.aux_head, nn.init.kaiming_normal_,
                self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                mode='fan_in', nonlinearity='relu')
        if self.use_graph:
            # Initialize graph processor weights
            for name, m in self.graph_processor.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            
            # Initialize fusion layer if it exists (for backward compatibility)
            # Note: In the new design, fusion_layer is created dynamically during forward pass
            if self.fusion_mode == 'concat' and hasattr(self, 'fusion_layer'):
                init_weight(self.fusion_layer, nn.init.kaiming_normal_,
                        self.norm_layer, cfg.bn_eps, cfg.bn_momentum,
                        mode='fan_in', nonlinearity='relu')
                
    def encode_decode(self, rgb, modal_x):
        # Get device from input tensor
        device = rgb.device
        orisize = rgb.shape
        
        # Get backbone features
        x = self.backbone(rgb, modal_x)
        
        # Process with graph-based enhancement if enabled
        if self.use_graph:
            # Get enhanced features from graph processor
            graph_features = self.graph_processor(rgb, modal_x)
            
            # Enhanced features can be used in place of or in addition to the last level features
            # Replace or enhance the last level features based on fusion mode
            if self.fusion_mode == 'weighted':
                # Weighted fusion of backbone and graph features
                x[-1] = (self.fusion_weight * x[-1] + (1 - self.fusion_weight) * graph_features).contiguous()
            elif self.fusion_mode == 'add':
                # Simple addition of backbone and graph features
                x[-1] = (x[-1] + graph_features).contiguous()
            else:
                # Default to concatenation - no direct replacement
                # For modalities like concat, custom handling would be implemented
                pass
        
        # Process with standard decoder using the enhanced features
        out = self.decode_head.forward(x)
        
        if isinstance(out, dict):  # For Mask2Former
            # Get mask predictions and resize
            masks = out['pred_masks']  # [B, num_queries, H, W]
            masks = F.interpolate(masks, size=orisize[2:], mode='bilinear', align_corners=False)
            # Apply sigmoid for final mask probabilities
            masks = masks.sigmoid()
            # Get class predictions
            logits = out['pred_logits']  # [B, num_queries, num_classes+1]
            
            # Return the dictionary format for loss computation
            return {
                'pred_logits': logits.to(device),
                'pred_masks': masks.to(device)
            }
        
        # For other decoders
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False).to(device)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False).to(device)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        
        if label is not None:
            if isinstance(self.criterion, tuple):
                loss = self.criterion[0](out, label.long()) + 0.2 * self.criterion[1](out, label.long())
            else:
                loss = self.criterion(out, label.long())
            if self.aux_head:
                loss += self.aux_rate * self.criterion(aux_fm, label.long())
            return loss
        return out