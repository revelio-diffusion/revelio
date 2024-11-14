import os
import torch
from datasets import load_dataset
from tqdm import trange
from training.hooked_sd import HookedStableDiffusion
from training.config import SDSAERunnerConfig
import torchvision.transforms as transforms

def get_model_activations(model, inputs, cfg):
    module_name = cfg.module_name
    list_of_hook_locations = [(module_name)]

    latents = model.vae.encode(inputs['pixel_values']).latent_dist.mode()
    latents = latents * model.vae.config.scaling_factor

    encoder_hidden_states = model.text_encoder(inputs['input_ids'], return_dict=False)[0]
    timestep = cfg.timestep    
    noise = torch.randn_like(latents)
    t = torch.tensor(timestep, dtype=torch.long, device=model.model.device)
    noisy_latents = model.noise_scheduler.add_noise(latents, noise, t)

    activations = model.run_with_cache(
        list_of_hook_locations,
        sample=noisy_latents,
        timestep=timestep,
        encoder_hidden_states = encoder_hidden_states.repeat(latents.size(0),1,1)
    )[1][(module_name)]
    return activations

        
@torch.inference_mode()
def get_feature_data(
    cfg: SDSAERunnerConfig,
    model: HookedStableDiffusion,
    max_number_of_images_per_iteration: int = 16_384,
):
    torch.cuda.empty_cache()
        
    dataset = load_dataset(cfg.dataset_path, split="train") 
    
    if cfg.dataset_path=="cifar100": 
        image_key = 'img'
    else:
        image_key = 'image'

    number_of_images = len(dataset)

    number_of_images_processed = 0
    while number_of_images_processed < number_of_images:
        torch.cuda.empty_cache()
        try:
            images = dataset[number_of_images_processed:number_of_images_processed + max_number_of_images_per_iteration][image_key]
        except StopIteration:
            break
        get_all_model_activations(model, images, "", cfg, number_of_images_processed ) 
        number_of_images_processed += max_number_of_images_per_iteration
        
def get_all_model_activations(model, images, texts, cfg, number_of_images_processed):
    max_batch_size = cfg.max_batch_size
    number_of_mini_batches = len(images) // max_batch_size
    remainder = len(images) % max_batch_size

    save_path = f'caltech101/SD15/timestep_{cfg.timestep}/mid'

    preprocess = transforms.Compose(
        [
            transforms.Resize(cfg.image_size),
            transforms.CenterCrop(cfg.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    for mini_batch in trange(number_of_mini_batches, desc = "Dashboard: forward pass images through SD"):
        image_batch = images[mini_batch*max_batch_size : (mini_batch+1)*max_batch_size]
        image_batch = [preprocess(image).unsqueeze(0) for image in image_batch]  
        image_batch = torch.cat(image_batch, dim=0).to('cuda') 
        inputs = {} 
        inputs['pixel_values'] = image_batch
        inputs.update(model.processor.tokenizer('', padding=True, return_tensors='pt').to('cuda'))

        sae_batches = get_model_activations(model, inputs, cfg)
        torch.save(sae_batches, os.path.join(save_path, f"image_{number_of_images_processed}_{number_of_images_processed +max_batch_size-1}_features.pt"))
        number_of_images_processed += max_batch_size

    if remainder>0:
        image_batch = images[-remainder:]
        image_batch = [preprocess(image).unsqueeze(0) for image in image_batch]  
        image_batch = torch.cat(image_batch, dim=0).to('cuda') 
        inputs = {} 
        inputs['pixel_values'] = image_batch
        inputs.update(model.processor.tokenizer('', padding=True, return_tensors='pt').to('cuda'))

        sae_batches = get_model_activations(model, inputs, cfg)
        torch.save(sae_batches, os.path.join(save_path, f"image_{number_of_images_processed}_{number_of_images_processed +remainder-1}_features.pt"))
        number_of_images_processed += remainder
    return sae_batches
