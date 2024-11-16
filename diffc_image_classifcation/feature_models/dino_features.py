import torch
import torch.nn.functional as F

from ezcolorlog import root_logger as logger
from transformers import Dinov2Model, AutoImageProcessor, Dinov2Config

from .base_encoder import BaseVisionTower


def extract_res_interp(model_name):
    valid_model_prefixes = [
        "facebook/dinov2-small",
        "facebook/dinov2-base",
        "facebook/dinov2-large",
        "facebook/dinov2-giant-imagenet1k-1-layer",
        "facebook/dinov2-giant",
    ]

    for prefix in valid_model_prefixes:
        if model_name.startswith(prefix):
            base_model_name = prefix
            break
    else:
        raise ValueError(f"Unknown vision tower: {model_name}")

    res = None
    interp = None

    parts = model_name[len(base_model_name) :].split("-")
    for part in parts:
        if part.startswith("res"):
            res = int(part[3:])
        elif part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, res, interp


class DinoVisionTower(BaseVisionTower):
    def __init__(self, vision_tower, config, delay_load=False):
        super(DinoVisionTower, self).__init__(vision_tower, config, delay_load)

        self._config = config  # Assign the config dictionary

        # Extract image resolution from the model name
        base_model_name, res, interp = extract_res_interp(self.vision_tower_name)
        self._vision_tower_name = vision_tower
        self.vision_tower_name = base_model_name
        self._image_size = res
        self._interp_size = interp
        self._patch_size = 14  # default patch size

        self.unfreeze_mm_vision_tower = config.get("unfreeze_mm_vision_tower", False)
        self.select_feature = config.get("mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Dinov2Config.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.vision_tower = Dinov2Model.from_pretrained(self.vision_tower_name)
        self.vision_tower._no_split_modules = ["Dinov2SwiGLUFFN"]

        _image_size = self.vision_tower.config.image_size
        if self._image_size is None:
            self._image_size = _image_size
        else:
            logger.warning(
                f"Overriding DinoVisionTower image size of {_image_size} with {self._image_size}"
            )

        shortest_edge = self._image_size

        processor = AutoImageProcessor.from_pretrained(
            self.vision_tower_name,
            crop_size=dict(height=self._image_size, width=self._image_size),
            size=dict(shortest_edge=shortest_edge),
        )

        logger.info(f"Dino Vision Processor: {processor}")
        self.image_processor = processor

        self._hidden_size = (
            self.vision_tower.embeddings.patch_embeddings.projection.out_channels
        )
        self._patch_size = (
            self.vision_tower.embeddings.patch_embeddings.projection.stride[0]
        )

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

    @property
    def image_size(self):
        return self._image_size

    def feature_select(self, outputs):
        sequence_output = outputs["last_hidden_state"]

        if self.select_feature == "cls_patch":
            image_features = sequence_output
        elif self.select_feature == "patch":
            image_features = sequence_output[:, 1:]
        elif self.select_feature == "cls":
            image_features = sequence_output[:, 0]
        else:
            raise ValueError(f"Unexpected select feature: {self.select_feature}")
        return image_features

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size**0.5)
            h = w = int(num_tokens**0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).to(image_features.dtype)

            image_features = image_features.permute(0, 2, 3, 1).contiguous()
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images):
        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower.forward(
                images.to(device=self.device, dtype=self.dtype)
            )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            interp_features = self.interpolate(image_features)
            return interp_features

    @property
    def num_patches_per_side(self):
        return int(self.num_patches**0.5)

    @property
    def num_patches(self):
        if self._interp_size is None:
            return (self._image_size // self._patch_size) ** 2
        else:
            return self._interp_size


class DINOFeatures:
    def __init__(self, config):
        self.config = config
        self._init_feature_encoder()

    def _init_feature_encoder(self):
        vision_tower_name = self.config["model_name"]
        self.model = DinoVisionTower(vision_tower_name, self.config).to("cuda")
        self.preprocess = self.model.image_processor

    def get_features(self, batch_images):
        image_features = self.model(batch_images)
        return image_features
