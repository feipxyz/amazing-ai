# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging
import numpy as np
import cv2
from typing import Callable, List, Union, Optional
import torch

# from panopticapi.utils import rgb2id

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from .aug import (
    RandomExpandAndCrop
)

class AugInput(T.AugInput):
    def __init__(
        self,
        image: np.ndarray,
        *,
        boxes: Optional[np.ndarray] = None,
        sem_seg: Optional[np.ndarray] = None,
        keypoints: Optional[np.ndarray] = None
    ):
        """
        Args:
            image (ndarray): (H,W) or (H,W,C) ndarray of type uint8 in range [0, 255], or
                floating point in range [0, 1] or [0, 255]. The meaning of C is up
                to users.
            boxes (ndarray or None): Nx4 float32 boxes in XYXY_ABS mode
            sem_seg (ndarray or None): HxW uint8 semantic segmentation mask. Each element
                is an integer label of pixel.
        """
        # _check_img_dtype(image)
        self.image = image
        self.boxes = boxes
        self.sem_seg = sem_seg
        self.keypoints = keypoints



def cv2_read_image(file_name, format):
    image = cv2.imread(file_name)
    if format == "RGB":
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class MobilePoseDatasetMapper:
    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        keypoint_hflip_indices: Optional[np.ndarray] = None
    ) -> None:

        self.image_format = image_format
        self.augmentations= T.AugmentationList(augmentations)
        self.is_train = is_train
        self.keypoint_hflip_indices = keypoint_hflip_indices

    @classmethod
    def from_config(cls, cfg):
        augs = [
            RandomExpandAndCrop((0.15, 0.3))
        ]

        ret = {
            "is_train": True,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret


    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # ---read image---
        # image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        image = cv2_read_image(dataset_dict["file_name"], self.image_format)
        utils.check_image_size(dataset_dict, image)

        # ---aug image---
        aug_input = AugInput(image, keypoints=np.array(dataset_dict["keypoints"]))
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        print(dataset_dict["image"].shape)

        # aug keypoint
        image_shape = image.shape[:2]  # h, w
        keypoints = utils.transform_keypoint_annotations(
            dataset_dict["keypoints"], 
            transforms, 
            image_shape, 
            keypoint_hflip_indices=self.keypoint_hflip_indices
        )
        dataset_dict["keypoints"] = torch.from_numpy(keypoints)

        return dataset_dict

