
from typing import Callable, Dict, List, Optional, Tuple, Union
import torch.nn as nn
import torch
import dsntnn

from aalib.layers import ShapeSpec
from aalib.structures import ImageList
from aalib.utils.registry import Registry
from aalib.config import configurable

from ..backbone import (
    Backbone,
    build_backbone,
) 

from .build import META_ARCH_REGISTRY


UPSAMPLE_MODULE_REGISTRY = Registry("UPSAMPLE_UNIT")
UPSAMPLE_MODULE_REGISTRY.__doc__ = """
upsample unit
"""

@META_ARCH_REGISTRY.register()
class CoordDsntnnNet(nn.Module):

    @configurable
    def __init__(
        self, 
        *,
        backbone: Backbone,
        compress_feature_name,
        compress_module: nn.Module,
        upsample_module: nn.Module,
        head: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float]
    ):
        super().__init__()
        self.backbone = backbone
        self.compress_feature_name = compress_feature_name
        self.compress_module = compress_module 
        self.upsample_module = upsample_module
        self.head = head
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        ch_num = list(backbone.output_shape().values())[0].channels
        compress_module = nn.Conv2d(ch_num, cfg.MODEL.COORDDSNTNN.COMPRESS_CHANNEL, 1, 1, 0, bias=False)
        upsample_module = build_upsample_module(cfg, ShapeSpec(channels=cfg.MODEL.COORDDSNTNN.COMPRESS_CHANNEL))
        ch_num = upsample_module.output_shape().channels
        head = nn.Conv2d(ch_num, cfg.MODEL.COORDDSNTNN.COORD_NUM, kernel_size=1, bias=False)

        return {
            "backbone": backbone,
            "compress_feature_name": cfg.MODEL.COORDDSNTNN.COMPRESS_FEATURE_NAME,
            "compress_module": compress_module,
            "upsample_module": upsample_module,
            "head": head,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD
        }

    @property
    def device(self):
        return self.pixel_mean.device


    # def forward(self, images):
    def forward(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        # 1. features
        features = self.backbone(images.tensor)
        features = features[self.compress_feature_name]
        # 2. compress channel
        features = self.compress_module(features)
        # 3. unsample
        features = self.upsample_module(features)
        # 4. Use a 1x1 conv to get one unnormalized heatmap per location
        heatmaps = self.head(features)
        # 5. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(heatmaps)
        # 6. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        if self.training:
            # Per-location euclidean losses
            euc_losses = dsntnn.euclidean_losses(coords, batched_inputs["keypoints"])
            # Per-location regularization losses
            reg_losses = dsntnn.js_reg_losses(heatmaps, batched_inputs["keypoints"], sigma_t=1.0)
            # Combine losses into an overall loss
            loss = dsntnn.average_loss(euc_losses + reg_losses)

            del images, coords, heatmaps

            loss = {
                "euc_loss": dsntnn.average_loss(euc_losses),
                "reg_loss": dsntnn.average_loss(reg_losses),
                "loss": loss
            }

            return loss
        
        else:
            return coords, heatmaps


class DUC(nn.Module):
    '''
    Initialize: inplanes, planes, upscale_factor
    OUTPUT: (planes // upscale_factor^2) * ht * wd
    '''

    def __init__(
        self, inplanes, planes, upscale_factor=2):
        super(DUC, self).__init__()
        self.conv = nn.Conv2d(
            inplanes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pixel_shuffle(x)
        return x


@UPSAMPLE_MODULE_REGISTRY.register()
class UpsamplePSF(nn.Module):
    @configurable
    def __init__(
        self, 
        ducs
    ):
        super().__init__()
        body = []
        for in_channels, out_channels, factor in ducs:
            body.append(DUC(in_channels, out_channels, factor))
        self.up_body = nn.Sequential(*body)
        self.out_channels = ducs[-1][1] // (ducs[-1][2] ** 2)

    @classmethod
    def from_config(cls, cfg, input_shape: ShapeSpec):
        return {
            "ducs": cfg.MODEL.PIXEL_SHUFFLE.DUCS,
        }

    def forward(self, x):
        return self.up_body(x)

    def output_shape(self):
        return ShapeSpec(channels=self.out_channels)
 

def build_upsample_module(cfg, input_shape):
    name = cfg.MODEL.COORDDSNTNN.UPSAMPLE_MODULE_NAME
    module = UPSAMPLE_MODULE_REGISTRY.get(name)(cfg, input_shape)
    return module


