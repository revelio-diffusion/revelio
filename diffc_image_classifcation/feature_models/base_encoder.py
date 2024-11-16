from abc import abstractmethod
import torch
import torch.nn as nn

from ezcolorlog import root_logger as logger


class ProcessorWrapper:
    def __init__(
        self,
        transform,
        height=378,
        width=378,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
    ):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors="pt"):
        output = {}
        output["pixel_values"] = [self._transforms(image)]
        return output


class BaseVisionTower(nn.Module):
    def __init__(self, vision_tower_name, config, delay_load=False):
        super(BaseVisionTower, self).__init__()

        self.is_loaded = False
        self._config = config  # Use _config to avoid conflicts

        self.vision_tower_name = vision_tower_name
        self.select_layer = config.get("mm_vision_select_layer", -1)
        self.select_feature = config.get("mm_vision_select_feature", "patch")
        self.unfreeze_mm_vision_tower = config.get("unfreeze_mm_vision_tower", False)
        logger.warning(f"Unfreezing MM Vision Tower: {self.unfreeze_mm_vision_tower}")
        self.delay_load = delay_load

    @abstractmethod
    def load_model(self, device_map=None):
        raise NotImplementedError("Subclasses must implement load_model")

    @abstractmethod
    def _forward(self, images):
        raise NotImplementedError("Subclasses must implement _forward")

    def forward(self, images):
        if isinstance(images, list):
            image_features = [self._forward(image.unsqueeze(0)) for image in images]
        else:
            image_features = self._forward(images)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        if hasattr(self.vision_tower, "dtype"):
            return self.vision_tower.dtype
        else:
            params = list(self.vision_tower.parameters())
            return params[0].dtype if params else torch.float32

    @property
    def device(self):
        if hasattr(self.vision_tower, "device"):
            return self.vision_tower.device
        else:
            params = list(self.vision_tower.parameters())
            return params[0].device if params else torch.device("cpu")

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        try:
            return self.config.hidden_size
        except AttributeError:
            return self._hidden_size

    @property
    def image_size(self):  # resolution
        try:
            return self.config.image_size
        except AttributeError:
            return self._image_size

    @property
    def patch_size(self):
        try:
            return self.config.patch_size
        except AttributeError:
            return self._patch_size

    @property
    def num_patches_per_side(self):
        if self._interp_size is not None:
            return int(self._interp_size**0.5)
        try:
            return self.image_size // self.patch_size
        except AttributeError:
            return self._num_patches_per_side

    @property
    def num_patches(self):
        if self._interp_size is not None:
            return self._interp_size
        try:
            return self.num_patches_per_side**2
        except AttributeError:
            return self._num_patches
