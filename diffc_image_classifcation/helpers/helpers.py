from feature_models.feature_extractor import FeatureExtractor


def clean_model_name(model_name):
    return (
        model_name.replace("/", "-")
        .replace(":", "-")
        .replace(".", "-")
        .replace(" ", "-")
        .replace("_", "-")
    )


def get_sample_feature(image, config):
    image = image.unsqueeze(0)
    feature_extractor = FeatureExtractor(config)
    image_features = feature_extractor.get_raw_features(
        image, None, config["diffusion_timestep"]
    )
    layer_idx = config["diffusion_layer"]
    layer = layer_idx.split(":")[0]
    idx = int(layer_idx.split(":")[1])
    return image_features[layer][idx]


def get_sample_feature_dit(image, config):
    image = image.unsqueeze(0)
    feature_extractor = FeatureExtractor(config)
    image_features = feature_extractor.get_raw_features(
        image, None, config["diffusion_timestep"]
    )
    layer_idx = config["diffusion_layer"]
    layer_idx = int(layer_idx)
    # layer = layer_idx.split(":")[0]
    # idx = int(layer_idx.split(":")[1])
    return image_features[:, layer_idx].mean(dim=0).unsqueeze(0)


def set_random_seed(seed):
    import random
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
