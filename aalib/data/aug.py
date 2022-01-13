import numpy as np

from detectron2.data.transforms import (
    Augmentation,
    Transform
)

class Crop(Transform):
    def __init__(self, x1, y1, x2, y2):
        self._set_attributes(locals())

    def apply_image(self, img: np.ndarray):
        img = img[self.y1 : self.y2, self.x1 : self.x2, :]
        return img
    
    def apply_coords(self, coords: np.ndarray):
        coords -= np.array([self.x1, self.y1])
        return coords

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def apply_polygons(self, polygons: list) -> list:
        raise NotImplementedError

    def inverse(self) -> "Transform":
        raise NotImplementedError

class RandomExpandAndCrop(Augmentation):

    def __init__(self, scale):
        if isinstance(scale, (float, int)):
            scale = (scale, scale)
        self._init(locals())


    def get_transform(self, image, keypoints):
        keypoints = np.array(keypoints).reshape((-1, 3))
        xmin = np.min(keypoints[:, 0])
        ymin = np.min(keypoints[:, 1])
        xmax = np.max(keypoints[:, 0])
        ymax = np.max(keypoints[:, 1])
        width = xmax - xmin
        height = ymax - ymin
        image_height, image_width = image.shape[0:2]

        scale_l = np.random.uniform(self.scale[0], self.scale[1])
        scale_t = np.random.uniform(self.scale[0], self.scale[1])
        scale_r = np.random.uniform(self.scale[0], self.scale[1])
        scale_b = np.random.uniform(self.scale[0], self.scale[1])
        x1 = np.clip(xmin - scale_l * width, 0, image_width - 1).astype(np.int32).item()
        y1 = np.clip(ymin - scale_t * height, 0, image_height - 1).astype(np.int32).item()
        x2 = np.clip(xmax + scale_r * width, 0, image_width - 1).astype(np.int32).item()
        y2 = np.clip(ymax + scale_b * height, 0, image_height - 1).astype(np.int32).item()
        print(x1, y1, x2, y2)

        return Crop(x1, y1, x2, y2)

