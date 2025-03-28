import torch
import torch.nn as nn
import torchvision.models as models

from ..net_utils import FeatureRectifyModule as FRM
from ..net_utils import FeatureFusionModule as FFM
from ..net_utils import ImprovedFeatureRectifyModule as IFRM
from ..net_utils import ImprovedFeatureFusionModule as IFFM

from collections import OrderedDict
import time
from engine.logger import get_logger

logger = get_logger()

class DualResNet(nn.Module):
    def __init__(self, backbone, pretrained=None, norm_fuse=nn.BatchNorm2d):
        super(DualResNet, self).__init__()
        
        # RGB stream
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.backbone = models.resnet152(pretrained=pretrained)
        else:
            raise ValueError("Unsupported backbone")
        
        # Depth stream (initialize with the same weights as RGB stream)
        if backbone == 'resnet50':
            self.backbone_d = models.resnet50(pretrained=pretrained)
        elif backbone == 'resnet101':
            self.backbone_d = models.resnet101(pretrained=pretrained)
        elif backbone == 'resnet152':
            self.backbone_d = models.resnet152(pretrained=pretrained)
        
        # Remove the final FC layer
        self.backbone.fc = nn.Identity()
        self.backbone_d.fc = nn.Identity()
        
        # Feature Rectify Modules
        self.frm1 = FRM(dim=256, reduction=1)
        self.frm2 = FRM(dim=512, reduction=1)
        self.frm3 = FRM(dim=1024, reduction=1)
        self.frm4 = FRM(dim=2048, reduction=1)
        
        # Feature Fusion Modules
        self.ffm1 = FFM(dim=256, reduction=1, num_heads=4, norm_layer=norm_fuse)
        self.ffm2 = FFM(dim=512, reduction=1, num_heads=8, norm_layer=norm_fuse)
        self.ffm3 = FFM(dim=1024, reduction=1, num_heads=16, norm_layer=norm_fuse)
        self.ffm4 = FFM(dim=2048, reduction=1, num_heads=32, norm_layer=norm_fuse)

    def forward(self, x_rgb, x_d):
                
        # RGB stream
        x_rgb = self.backbone.conv1(x_rgb)
        x_rgb = self.backbone.bn1(x_rgb)
        x_rgb = self.backbone.relu(x_rgb)
        x_rgb = self.backbone.maxpool(x_rgb)

        # Depth stream
        x_d = self.backbone_d.conv1(x_d)
        x_d = self.backbone_d.bn1(x_d)
        x_d = self.backbone_d.relu(x_d)
        x_d = self.backbone_d.maxpool(x_d)

        # Layer 1
        x_rgb = self.backbone.layer1(x_rgb)
        x_d = self.backbone_d.layer1(x_d)
        x_rgb, x_d = self.frm1(x_rgb, x_d)
        out1 = self.ffm1(x_rgb, x_d)

        # Layer 2
        x_rgb = self.backbone.layer2(x_rgb)
        x_d = self.backbone_d.layer2(x_d)
        x_rgb, x_d = self.frm2(x_rgb, x_d)
        out2 = self.ffm2(x_rgb, x_d)

        # Layer 3
        x_rgb = self.backbone.layer3(x_rgb)
        x_d = self.backbone_d.layer3(x_d)
        x_rgb, x_d = self.frm3(x_rgb, x_d)
        out3 = self.ffm3(x_rgb, x_d)

        # Layer 4
        x_rgb = self.backbone.layer4(x_rgb)
        x_d = self.backbone_d.layer4(x_d)
        x_rgb, x_d = self.frm4(x_rgb, x_d)
        out4 = self.ffm4(x_rgb, x_d)

        return (out1, out2, out3, out4)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            load_dualpath_model(self, pretrained)
        elif pretrained is None:
            pass  # Use default initialization
        else:
            raise TypeError('pretrained must be a str or None')

def load_dualpath_model(model, model_file, is_restore=False):
    # load raw state_dict
    t_start = time.time()
    if isinstance(model_file, str):
        raw_state_dict = torch.load(model_file, map_location=torch.device('cpu'))
        if 'model' in raw_state_dict.keys():
            raw_state_dict = raw_state_dict['model']
    else:
        raw_state_dict = model_file
    
    # copy to depth backbone
    state_dict = {}
    for k, v in raw_state_dict.items():
        if k.startswith('backbone.'):
            state_dict[k] = v
            state_dict[k.replace('backbone.', 'backbone_d.')] = v
        else:
            state_dict[k] = v

    t_ioend = time.time()

    if is_restore:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = 'module.' + k
            new_state_dict[name] = v
        state_dict = new_state_dict

    model.load_state_dict(state_dict, strict=False)
    
    del state_dict
    t_end = time.time()
    logger.info(
        "Load model, Time usage:\n\tIO: {}, initialize parameters: {}".format(
            t_ioend - t_start, t_end - t_ioend))

    return model

class dual_resnet50(DualResNet):
    def __init__(self, pretrained=True, norm_fuse=nn.BatchNorm2d, **kwargs):
        super(dual_resnet50, self).__init__(backbone='resnet50', pretrained=pretrained, norm_fuse=norm_fuse, **kwargs)

class dual_resnet101(DualResNet):
    def __init__(self, pretrained=True, norm_fuse=nn.BatchNorm2d, **kwargs):
        super(dual_resnet101, self).__init__(backbone='resnet101', pretrained=pretrained, norm_fuse=norm_fuse, **kwargs)

class dual_resnet152(DualResNet):
    def __init__(self, pretrained=True, norm_fuse=nn.BatchNorm2d, **kwargs):
        super(dual_resnet152, self).__init__(backbone='resnet152', pretrained=pretrained, norm_fuse=norm_fuse, **kwargs)
