import torch
import torch.nn as nn

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict

from .position_encoding import PositionEmbeddingSine

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, d_model:int, name:str, backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        self.proj = nn.Conv2d(num_channels, d_model, 1)
        self.position_embedding = PositionEmbeddingSine(d_model//2, normalize=True)

    def forward(self, xs, m=None, end_points=None):
        if end_points is None:
            end_points = {}
        x = self.body(xs)['0']
        if m is None:
            m = torch.ones([x.shape[0], x.shape[2], x.shape[3]], dtype=bool, device=x.device)
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x = self.proj(x)
        pos = self.position_embedding(x, mask)
        B, C, W, H = x.shape
        end_points['image_feature'] = x.view(B, C, -1)
        end_points['img_mask'] = mask.view(B, -1)
        end_points['img_pos'] = pos.view(B, C, -1)
        return end_points


class VisualBackbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, d_model, name='resnet34', return_interm_layers=False, dilation=False):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=True, norm_layer=FrozenBatchNorm2d)
            # pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = backbone.fc.in_features
        super().__init__(d_model, name, backbone, num_channels, return_interm_layers)




if __name__ == '__main__':
    model = VisualBackbone(288).cuda()
    image = torch.rand([2, 3, 1200, 1920]).cuda()
    mask = torch.ones([2, 1200, 1920]).bool().cuda()
    out = model(image)
    # print(out.shape)
