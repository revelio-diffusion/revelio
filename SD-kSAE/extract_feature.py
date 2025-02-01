from training.hooked_sd import HookedStableDiffusion
import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import Tensor, topk
from torchvision import datasets
from tqdm import tqdm
from training.config import SDSAERunnerConfig
import torchvision.transforms as transforms
from PIL import Image
from typing import List, Any
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, dataset, image_key: str, image_size: int):
        self.dataset = dataset
        self.image_key = image_key
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Load the image from the dataset
        raw_image = self.dataset[idx][self.image_key].convert("RGB")
        
        # Preprocess the image into a tensor
        processed_image = self.preprocess(raw_image)

        # Return the processed image
        return processed_image

def process_batches(
    model: HookedStableDiffusion,
    dataset: List[Any],
    cfg: SDSAERunnerConfig,
) -> None:
    """Process dataset in batches and save activations."""
    image_processsed = 0
    os.makedirs(cfg.save_path, exist_ok=True)

    # Wrap the dataset with the ImageDataset class
    processed_dataset = ImageDataset(dataset, cfg.image_key, cfg.image_size)
    dataloader = DataLoader(processed_dataset, batch_size=cfg.max_batch_size, shuffle=False)

    caption = ''    # empty prompt
    for i, image_batch in enumerate(tqdm(dataloader, desc='Extracting features')):
        try:
            inputs = {
                'pixel_values': image_batch.to('cuda'),
                'input_ids': model.tokenizer(
                    caption, max_length=model.tokenizer.model_max_length, padding="max_length", return_tensors='pt').input_ids.to('cuda')
            }
            activations = get_model_activations(model, inputs, cfg)
            torch.save(activations, os.path.join(cfg.save_path, f"image_{image_processsed}_{image_processsed +len(image_batch)-1}_features.pt"))
            logging.info(f"Batch {i} processed and saved.")
            image_processsed += len(image_batch)
        except Exception as e:
            logging.error(f"Error processing batch {i}: {str(e)}")
            continue


def get_model_activations(model, inputs, cfg):
    """Extract activations from the model."""
    latents = model.vae.encode(inputs['pixel_values']).latent_dist.mode()
    latents = latents * model.vae.config.scaling_factor

    encoder_hidden_states = model.text_encoder(inputs['input_ids'], return_dict=False)[0]
    noise = torch.randn_like(latents)
    t = torch.tensor(cfg.timestep, dtype=torch.long, device=model.model.device)
    noisy_latents = model.noise_scheduler.add_noise(latents, noise, t)

    activations = model.run_with_cache(
        [(cfg.block_name)],
        sample=noisy_latents,
        timestep=cfg.timestep,
        encoder_hidden_states = encoder_hidden_states.repeat(latents.size(0),1,1)
    )[1][(cfg.block_name)]
    return activations


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Extract activations from a Stable Diffusion model using a given dataset.")
    parser.add_argument("--image_size", type=int, default=256, help="Size of input images.")
    parser.add_argument("--model_name", type=str, default="runwayml/stable-diffusion-v1-5", help="Name of the diffusion model to use.")
    parser.add_argument("--timestep", type=int, default=25, help="Timestep for the diffusion process.")
    parser.add_argument("--block_name", type=str, default="mid_block", help="Which block to extract activations from. Options: 'mid_block' (bottleneck), 'up_blocks.0' (up_ft0), 'up_blocks.1' (up_ft1), 'up_blocks.2' (up_ft2)")
    parser.add_argument("--image_key", type=str, default="image")
    parser.add_argument("--dataset_name", type=str, default="dpdl-benchmark/caltech101", help="Dataset to use for feature extraction. Options: 'timm/oxford-iiit-pet' (OxfordPet), 'evanarlian/imagenet_1k_resized_256' (ImageNet)")
    parser.add_argument("--max_batch_size", type=int, default=32, help="Maximum batch size to save.")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save extracted features. If not specified, it will be generated automatically.")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    dataset_short = args.dataset_name.split('/')[-1]
    if args.save_path is None:
        model_short = args.model_name.split('/')[-1].replace('stable-diffusion-', 'SD')
        args.save_path = f"{model_short}/{model_short}/timestep_{args.timestep}/{args.block_name}"

    cfg = SDSAERunnerConfig(
        image_size=args.image_size,
        model_name=args.model_name,
        timestep=args.timestep,
        block_name=args.block_name,
        image_key=args.image_key,
        dataset_name=args.dataset_name,
        max_batch_size=args.max_batch_size,
        save_path=args.save_path,
        device=device,
    )

    model = HookedStableDiffusion(cfg.model_name, cfg.image_size, cfg.device)   # load model
    model.eval()
    dataset = load_dataset(cfg.dataset_name, split="train") 
    process_batches(model, dataset, cfg)
