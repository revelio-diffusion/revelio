from torchvision import transforms


IMAGE_SIZE = 512


model_base_dict = {
    "openai/clip-vit-large-patch14-336": "clip",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "clip",
    "openai/clip-resnet-50": "clip",
    "stabilityai/stable-diffusion-2-1": "diffusion",
    "facebook/dinov2-large": "dino",
    "stable-diffusion-v1-5/stable-diffusion-v1-5": "diffusion",
    "stabilityai/stable-diffusion-2-1-base": "diffusion",
    "stabilityai/stable-diffusion-xl-base-1.0": "dit",
    "facebook/DiT-XL-2-256": "dit",
    "facebook/DiT-XL-2-512": "dit",
}  # to instantiate the FeatureExtractor class in the main


diffusion_transformers_val = transforms.Compose(
    [
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

diffusion_transformers_train = transforms.Compose(
    [
        # Data augmentation: RandomResizedCrop and RandomHorizontalFlip
        transforms.RandomResizedCrop(
            IMAGE_SIZE, scale=(0.8, 1.0)
        ),  # Random crop with scaling
        transforms.RandomHorizontalFlip(
            p=0.5
        ),  # Random horizontal flip with 50% probability
        # Convert to tensor and normalize
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        ),  # Normalize the pixel values
    ]
)


clip_transforms = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ]
)
