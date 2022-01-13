# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import unittest
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import aalib.export.torchscript  # apply patch # noqa
# from aalib import model_zoo
from aalib.config import get_cfg
from aalib.layers import ShapeSpec
# from aalib.modeling.backbone import build_resnet_backbone
# from aalib.modeling.backbone.fpn import build_resnet_fpn_backbone


# class TestBackBone(unittest.TestCase):
#     def test_resnet_scriptability(self):
#         cfg = get_cfg()
#         resnet = build_resnet_backbone(cfg, ShapeSpec(channels=3))

#         scripted_resnet = torch.jit.script(resnet)

#         inp = torch.rand(2, 3, 100, 100)
#         out1 = resnet(inp)["res4"]
#         out2 = scripted_resnet(inp)["res4"]
#         self.assertTrue(torch.allclose(out1, out2))

#     def test_fpn_scriptability(self):
#         cfg = model_zoo.get_config("Misc/scratch_mask_rcnn_R_50_FPN_3x_gn.yaml")
#         bb = build_resnet_fpn_backbone(cfg, ShapeSpec(channels=3))
#         bb_s = torch.jit.script(bb)

#         inp = torch.rand(2, 3, 128, 128)
#         out1 = bb(inp)["p5"]
#         out2 = bb_s(inp)["p5"]
#         self.assertTrue(torch.allclose(out1, out2))

#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.

"""
Panoptic-DeepLab Training Script.
This script is a simplified version of the training script in detectron2/tools.
"""

import os
import torch
from aalib.engine import default_argument_parser

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
#     default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    from aalib.modeling import build_model
    model = build_model(cfg)
    model.eval()
    x = torch.randn((3, 512, 512))
    batch_input = [{"image": x}]
    y = model(batch_input)
    print(y)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
    