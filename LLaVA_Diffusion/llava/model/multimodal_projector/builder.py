import torch
import torch.nn as nn
import re
import torch.nn.functional as F


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": "identity"}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels), nn.GELU(), nn.Linear(channels, channels)
        )

    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == "identity":
        return IdentityMap()

    raise ValueError(f"Unknown projector type: {projector_type}")


class DiffusionFeatureClassifier_V4(nn.Module):
    def __init__(self, input_channels=1280, dropout_rate=0.0):
        super(DiffusionFeatureClassifier_V4, self).__init__()

        # First convolutional layer (maintain spatial size)
        self.conv1 = nn.Conv2d(input_channels, 1024, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(1024)
        # self.dropout1 = nn.Dropout2d(p=dropout_rate)

        # First downsampling
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        # self.bn2 = nn.BatchNorm2d(1024)
        # self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.act = nn.GELU()

    def forward(self, x):
        # Input: [batch_size, 1280, H, W]

        # x = self.input_bn(x)  # Normalize input features from another network

        # First convolution
        x = self.conv1(x)  # [batch_size, 1024, H, W]
        # x = self.bn1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)
        x = self.act(x)

        # First downsampling
        x = self.conv2(x)  # [batch_size, 1024, H/2, W/2]
        # x = self.bn2(x)
        # x = F.relu(x)
        # x = self.dropout2(x)

        # reshape t0 be [bs, H*W, C]
        x = x.view(x.size(0), x.size(1), -1).permute(0, 2, 1)

        return x


def build_vision_projector_diffusion(config, delay_load=False, **kwargs):
    projector_type = getattr(config, "mm_projector_type", "linear")

    # if projector_type == 'linear':
    #     return nn.Linear(config.mm_hidden_size, config.hidden_size)

    # mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    # if mlp_gelu_match:
    #     mlp_depth = int(mlp_gelu_match.group(1))
    #     modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
    #     for _ in range(1, mlp_depth):
    #         modules.append(nn.GELU())
    #         modules.append(nn.Linear(config.hidden_size, config.hidden_size))
    #     return nn.Sequential(*modules)

    # if projector_type == 'identity':
    #     return IdentityMap()

    # raise ValueError(f'Unknown projector type: {projector_type}')

    conv_projector = DiffusionFeatureClassifier_V4()
    mlp_gelu_match = re.match(r"^mlp(\d+)x_gelu$", projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        # return nn.Sequential(*modules)

    # combine conv_projector and mlp_gelu_projector
    return nn.Sequential(conv_projector, nn.Sequential(*modules))
