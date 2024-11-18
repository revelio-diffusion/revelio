from ezcolorlog import root_logger as logger
from torch.utils.data import Dataset
import os
import json
from datasets import load_dataset


# Custom dataset to load images from Hugging Face dataset
class HuggingFaceImageDataset(Dataset):
    def __init__(self, hf_dataset, diffusion_preprocess=None, clip_preprocess=None):
        self.hf_dataset = hf_dataset
        self.diffusion_preprocess = diffusion_preprocess
        self.clip_preprocess = clip_preprocess
        try:
            self.num_classes = len(set(hf_dataset["label"]))
        except:
            self.num_classes = len(set(hf_dataset["variant"]))  # for fgvc dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["image"].convert("RGB")

        if self.diffusion_preprocess:
            diffusion_image = self.diffusion_preprocess(image)
        else:
            diffusion_image = None
        if self.clip_preprocess:
            clip_image = self.clip_preprocess(image)
        else:
            clip_image = None
        # Process label
        try:
            if isinstance(sample["label"], int):
                int_label = sample["label"]
            elif isinstance(sample["label"], list):
                int_label = sample["label"][0]
        except:
            if isinstance(sample["variant"], int):
                int_label = sample["variant"]
            elif isinstance(sample["variant"], list):
                int_label = sample["variant"][0]

        # image_name = f"image_{idx}"  # Use original index for uniqueness
        # return image, int_label, idx
        return diffusion_image, clip_image, int_label, idx


# Custom collate function to handle PIL images
def custom_collate_fn(batch):
    images, labels, paths = zip(*batch)
    return list(images), list(labels), list(paths)


# Function to create class index-to-name mapping
def create_class_idx_mapping(filtered_dataset):
    class_names = [
        sample["text"]
        for sample in filtered_dataset
        if isinstance(sample["label"], str)
    ]
    unique_class_names = list(set(class_names))
    class_idx_mapping = {name: idx for idx, name in enumerate(unique_class_names)}
    if logger:
        logger.info(
            f"Class index mapping created with {len(class_idx_mapping)} classes."
        )
    return class_idx_mapping


# Function to save the class index mapping
def save_class_idx_mapping(mapping, output_dir):
    mapping_path = os.path.join(output_dir, "class_idx_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    if logger:
        logger.info(f"Class index mapping saved to {mapping_path}")


# Function to load the dataset from Hugging Face
def load_huggingface_dataset(dataset_name, split):
    if logger:
        logger.info(f"Loading dataset {dataset_name} with split {split}")
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if logger:
        logger.info(f"Dataset loaded: {len(dataset)} samples")
    return dataset
