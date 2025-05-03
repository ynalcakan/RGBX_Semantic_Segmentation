import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.init_func import init_weight
from utils.load_utils import load_pretrain
from functools import partial
from utils.loss_opr import Mask2FormerLoss

from engine.logger import get_logger

logger = get_logger()

class EncoderDecoder(nn.Module):
    def __init__(self, cfg=None, criterion=nn.CrossEntropyLoss(reduction='mean', ignore_index=255), norm_layer=nn.BatchNorm2d):
        super(EncoderDecoder, self).__init__()
        self.channels = [64, 128, 320, 512]
        self.norm_layer = norm_layer
        self.cfg = cfg
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
            self.channels = [64, 128, 320, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b4':
            logger.info('Using backbone: Segformer-B4')
            from .encoders.dual_segformer import mit_b4 as backbone
            self.channels = [64, 128, 320, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b3':
            logger.info('Using backbone: Segformer-B3')
            from .encoders.dual_segformer import mit_b3 as backbone
            self.channels = [64, 128, 320, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b2':
            logger.info('Using backbone: Segformer-B2')
            from .encoders.dual_segformer import mit_b2 as backbone
            self.channels = [64, 128, 320, 512]
            self.backbone = backbone(norm_fuse=norm_layer)
        elif cfg.backbone == 'mit_b1':
            logger.info('Using backbone: Segformer-B1')
            from .encoders.dual_segformer import mit_b1 as backbone
            self.channels = [32, 64, 160, 256]
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

        if cfg.decoder == 'MLPDecoder':
            logger.info('Using MLP Decoder')
            from .decoders.MLPDecoder import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
            logger.info(f"Initialized decode_head with in_channels={self.channels}, embed_dim={cfg.decoder_embed_dim}")
        elif cfg.decoder == 'MLPDecoderpp':
            logger.info('Using MLP Decoderpp')
            from .decoders.MLPDecoderpp import DecoderHead
            self.decode_head = DecoderHead(in_channels=self.channels, num_classes=cfg.num_classes, norm_layer=norm_layer, embed_dim=cfg.decoder_embed_dim)
        
        elif cfg.decoder == 'UPernet':
            logger.info('Using Upernet Decoder')
            from .decoders.UPernet import UPerHead
            self.decode_head = UPerHead(in_channels=self.channels ,num_classes=cfg.num_classes, norm_layer=norm_layer, channels=512)
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

    def encode_decode(self, rgb, modal_x):
        orisize = rgb.shape
        x = self.backbone(rgb, modal_x)
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
                'pred_logits': logits,
                'pred_masks': masks
            }
        
        # For other decoders
        out = F.interpolate(out, size=orisize[2:], mode='bilinear', align_corners=False)
        if self.aux_head:
            aux_fm = self.aux_head(x[self.aux_index])
            aux_fm = F.interpolate(aux_fm, size=orisize[2:], mode='bilinear', align_corners=False)
            return out, aux_fm
        return out

    def forward(self, rgb, modal_x, label=None):
        if self.aux_head:
            out, aux_fm = self.encode_decode(rgb, modal_x)
        else:
            out = self.encode_decode(rgb, modal_x)
        if label is not None:
            if isinstance(self.criterion, tuple):
                # compute individual losses
                loss1 = self.criterion[0](out, label.long())
                loss2 = self.criterion[1](out, label.long())
                # apply user-configurable weights
                w1 = getattr(self.cfg, 'loss_weight1', 1.0)
                w2 = getattr(self.cfg, 'loss_weight2', 0.5)
                loss = w1 * loss1 + w2 * loss2
                return loss, {"loss1": loss1, "loss2": loss2}
            else:
                loss = self.criterion(out, label.long())
                return loss
        return out