from feature_models.clip_features import CLIPFeatures
from feature_models.dino_features import DINOFeatures
from feature_models.diffusion_features import DiffusionFeatures, DiffusionFeaturesDiT
import torch
from torch import nn


class FeatureExtractor(nn.Module):
    def __init__(self, config):
        super(FeatureExtractor, self).__init__()
        self.config = config
        self.feature_model_dict = {
            "clip": CLIPFeatures,
            "dino": DINOFeatures,
            "diffusion": DiffusionFeatures,
            "dit": DiffusionFeaturesDiT,
        }
        self.feature_model = self.feature_model_dict[config["feature_model"].lower()](
            config
        )
        self.preprocess = self.feature_model.preprocess
        self.device = torch.device("cpu")  # Placeholder; will be set by accelerator

        self._freeze_model()

    def _freeze_model(self):
        for param in self.feature_model.model.parameters():
            param.requires_grad = False

    def get_features(self, batch_images):
        return self.feature_model.get_features(batch_images)

    def get_raw_features(self, batch_images, prompt_embeds, time_step):
        return self.feature_model.get_raw_features(
            batch_images, prompt_embeds, time_step
        )
