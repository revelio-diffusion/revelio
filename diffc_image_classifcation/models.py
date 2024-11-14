import torch.nn as nn
import torch.nn.functional as F
from feature_models.feature_extractor import FeatureExtractor
import torch


class DiffusionFeatureClassifier_V4(nn.Module):
    def __init__(self, input_channels=1280, num_classes=37, dropout_rate=0.3):
        super(DiffusionFeatureClassifier_V4, self).__init__()

        # Input feature batch normalization (parameter-free)
        self.input_bn = nn.BatchNorm2d(input_channels, affine=False)

        # First convolutional layer (maintain spatial size)
        self.conv1 = nn.Conv2d(input_channels, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        # First downsampling
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # Second downsampling
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # Third downsampling
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Input: [batch_size, 1280, H, W]

        x = self.input_bn(x)  # Normalize input features from another network

        # First convolution
        x = self.conv1(x)  # [batch_size, 1024, H, W]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # First downsampling
        x = self.conv2(x)  # [batch_size, 1024, H/2, W/2]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Second downsampling
        x = self.conv3(x)  # [batch_size, 1024, H/4, W/4]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Third downsampling
        x = self.conv4(x)  # [batch_size, 1024, H/8, W/8]
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        # Global Average Pooling
        x = self.gap(x)  # [batch_size, 1024, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 1024]

        # Dropout and Fully Connected Layer
        x = self.dropout_fc(x)
        x = self.fc(x)  # [batch_size, num_classes]

        return x


class ImageClassifer(nn.Module):
    def __init__(self, config):
        super(ImageClassifer, self).__init__()
        self.config = config
        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_extractor.feature_model.model.to(self.config["device"])
        self.feature_model = self.feature_extractor.feature_model.model
        try:
            self.diffusion_pipe = (
                self.feature_extractor.feature_model.model.diffusion_pipe
            )
        except:
            self.dit_pipe = self.feature_extractor.feature_model.model.dit_pipe
        self.classifer = DiffusionFeatureClassifier_V4(
            self.config["input_channels"],
            self.config["num_classes"],
            self.config["dropout_rate"],
        )

        self.layer_idx = self.config["diffusion_layer"]
        self.layer = self.layer_idx.split(":")[0]
        try:
            self.idx = int(self.layer_idx.split(":")[1])
        except:
            self.idx = None

    def get_features(self, batch_images, prompt_embs, time_step):
        image_features = self.feature_extractor.get_raw_features(
            batch_images, prompt_embs, time_step
        )
        if self.idx is None:
            return image_features[:, int(self.layer)]
        return image_features[self.layer][self.idx]

    def forward(self, batch_images, prompt_embs, time_step):
        features = self.get_features(batch_images, prompt_embs, time_step)
        return self.classifer(features)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DiTClassifier_V4(nn.Module):
    def __init__(self, input_features=1280, num_classes=37, dropout_rate=0.3):
        super(DiTClassifier_V4, self).__init__()

        # Input feature batch normalization (parameter-free)
        self.input_bn = nn.BatchNorm2d(input_features, affine=False)

        # First convolutional layer (maintain spatial size)
        self.conv1 = nn.Conv2d(input_features, 1024, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        # First downsampling
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        # Second downsampling
        self.conv3 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.dropout3 = nn.Dropout2d(p=dropout_rate)

        # Third downsampling
        self.conv4 = nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(1024)
        self.dropout4 = nn.Dropout2d(p=dropout_rate)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Fully connected layer
        self.dropout_fc = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Input: [batch_size, H*W, 1152]

        # reshape to [batch_size, 1152, H, W]
        x = x.view(x.size(0), 1152, 32, 32)

        x = self.input_bn(x)  # Normalize input features from another network

        # First convolution
        x = self.conv1(x)  # [batch_size, 1024, H, W]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        # First downsampling
        x = self.conv2(x)  # [batch_size, 1024, H/2, W/2]
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Second downsampling
        x = self.conv3(x)  # [batch_size, 1024, H/4, W/4]
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        # Third downsampling
        x = self.conv4(x)  # [batch_size, 1024, H/8, W/8]
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        # Global Average Pooling
        x = self.gap(x)  # [batch_size, 1024, 1, 1]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, 1024]

        # Dropout and Fully Connected Layer
        x = self.dropout_fc(x)
        x = self.fc(x)  # [batch_size, num_classes]

        return x


class ImageClassifier_DiT(nn.Module):
    def __init__(self, config):
        super(ImageClassifier_DiT, self).__init__()
        self.config = config
        self.device = self.config.get("device", torch.device("cpu"))

        # Feature extractor initialization
        self.feature_extractor = FeatureExtractor(self.config)
        self.feature_extractor.feature_model.model.to(self.device)
        self.feature_model = self.feature_extractor.feature_model.model

        # Check if we are using diffusion pipeline or DiT pipeline
        self.pipe = getattr(
            self.feature_extractor.feature_model.model,
            "diffusion_pipe",
            getattr(self.feature_extractor.feature_model.model, "dit_pipe", None),
        )

        # Initialize classifier
        self.classifier = DiTClassifier_V4(
            input_features=1152,
            num_classes=self.config["num_classes"],
            dropout_rate=self.config["dropout_rate"],
        )

        # Parse the layer index
        self.layer_idx = self.config["diffusion_layer"]
        self.layer = self.layer_idx.split(":")[0]
        try:
            self.idx = int(self.layer_idx.split(":")[1])
        except (IndexError, ValueError):
            self.idx = None

    def get_features(self, batch_images, prompt_embs, time_step):
        image_features = self.feature_extractor.get_raw_features(
            batch_images, prompt_embs, time_step
        )

        # If no specific index is given, return the full feature map for the selected layer
        if self.idx is None:
            return image_features[:, int(self.layer), :]

        # Return the feature at the specific index within the selected layer
        return image_features[:, int(self.layer), :, self.idx]

    def forward(self, batch_images, prompt_embs, time_step):
        # Extract features from the specific layer
        features = self.get_features(batch_images, prompt_embs, time_step)
        return self.classifier(features)
