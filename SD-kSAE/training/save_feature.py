import json
import os
import torch
from datasets import load_dataset
from torch import topk
from tqdm import tqdm
from training.k_sparse_autoencoder import KSparseAutoencoder
from training.sd_activations_store import SDActivationsStore

def get_sae_activations(model_activaitons, k_sparse_autoencoder):
    hook_name = "hook_hidden_post"
    sae_activations = k_sparse_autoencoder.run_with_cache(model_activaitons)[1][hook_name]
    sae_activations = sae_activations.to(k_sparse_autoencoder.cfg.device)
    return sae_activations

        
def save_highest_activating_images_high(max_activating_image_indices, max_activating_image_values, directory, dataset, image_key, sae_mean_acts, sae_sparsity, max_activating_image_label_indices):
    assert max_activating_image_values.size() == max_activating_image_indices.size(), "size of max activating image indices doesn't match the size of max activing values."
    number_of_neurons, number_of_max_activating_examples = max_activating_image_values.size()

    num_activating = sae_sparsity * len(dataset)
    _, neurons = topk(sae_mean_acts * (num_activating > 10), k=100)
    
    for neuron in tqdm(neurons, desc = "saving highest activating images"):
        neuron_dead = True
        for max_activating_image in range(number_of_max_activating_examples):
            if max_activating_image_values[neuron, max_activating_image].item()>0:
                if neuron_dead:
                    if not os.path.exists(f"{directory}/{neuron}"):
                        os.makedirs(f"{directory}/{neuron}")
                    neuron_dead = False
                image = dataset[int(max_activating_image_indices[neuron, max_activating_image].item())][image_key]
                image.save(f"{directory}/{neuron}/{max_activating_image}_{int(max_activating_image_indices[neuron, max_activating_image].item())}_{max_activating_image_values[neuron, max_activating_image].item():.4g}_{int(max_activating_image_label_indices[neuron, max_activating_image].item())}.png", "PNG")

def get_new_top_k(first_values, first_indices, second_values, second_indices, k):
    total_values = torch.cat([first_values, second_values], dim = 1)
    total_indices = torch.cat([first_indices, second_indices], dim = 1)
    new_values, indices_of_indices = topk(total_values, k=k, dim=1)
    new_indices = torch.gather(total_indices, 1, indices_of_indices)
    return new_values, new_indices

@torch.inference_mode()
def get_feature_data(
    k_sparse_autoencoder: KSparseAutoencoder,
    activation_store: SDActivationsStore,
    number_of_images: int = 32_768,
    number_of_max_activating_images: int = 10,
    load_pretrained = False,
):

    torch.cuda.empty_cache()
    k_sparse_autoencoder.eval()
    
    dataset = load_dataset(k_sparse_autoencoder.cfg.dataset_path, split="train")

    if k_sparse_autoencoder.cfg.dataset_path=="cifar100": # Need to put this in the cfg
        image_key = 'img'
    else:
        image_key = 'image'
        
    directory = k_sparse_autoencoder.cfg.feature_dir
    if load_pretrained:
        max_activating_image_indices = torch.load(f'{directory}/max_activating_image_indices.pt')
        max_activating_image_values = torch.load(f'{directory}/max_activating_image_values.pt')
    else:
        max_activating_image_indices = torch.zeros([k_sparse_autoencoder.cfg.d_sae, number_of_max_activating_images]).to(k_sparse_autoencoder.cfg.device)
        max_activating_image_values = torch.zeros([k_sparse_autoencoder.cfg.d_sae, number_of_max_activating_images]).to(k_sparse_autoencoder.cfg.device)
        sae_sparsity = torch.zeros([k_sparse_autoencoder.cfg.d_sae]).to(k_sparse_autoencoder.cfg.device)
        sae_mean_acts = torch.zeros([k_sparse_autoencoder.cfg.d_sae]).to(k_sparse_autoencoder.cfg.device)
        number_of_images_processed = 0
        activations = activation_store.feature_dataset
        sorted_features = torch.stack([x for x in activations]).to('cuda')  # Stack features

        sae_activations = torch.nn.functional.relu(get_sae_activations(sorted_features, k_sparse_autoencoder)).transpose(0,1) # tensor of size [feature_idx, batch]
        sae_mean_acts += sae_activations.sum(dim = 1)
        sae_sparsity += (sae_activations>0).sum(dim = 1)
        
        values, indices = topk(sae_activations, k = number_of_max_activating_images, dim = 1) # sizes [sae_idx, images] is the size of this matrix correct?
        
        max_activating_image_values, max_activating_image_indices = get_new_top_k(max_activating_image_values, max_activating_image_indices, values, indices, number_of_max_activating_images)
        
        number_of_images_processed += len(dataset)
        
        sae_mean_acts /= (sae_sparsity +1e-8)
        sae_sparsity /= number_of_images_processed
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        max_activating_image_label_indices = torch.tensor([dataset[int(index)]['label'] for index in tqdm(max_activating_image_indices.flatten(), desc = "getting image labels")])
        max_activating_image_label_indices = max_activating_image_label_indices.view(max_activating_image_indices.shape)
        torch.save(max_activating_image_indices, f'{directory}/max_activating_image_indices.pt')
        torch.save(max_activating_image_values, f'{directory}/max_activating_image_values.pt')
        torch.save(max_activating_image_label_indices, f'{directory}/max_activating_image_label_indices.pt')
        torch.save(sae_sparsity, f'{directory}/sae_sparsity.pt')
        torch.save(sae_mean_acts, f'{directory}/sae_mean_acts.pt')
        num_activating = sae_sparsity * len(dataset)

        _, neurons = topk(sae_mean_acts * (num_activating > 10), k=sae_mean_acts.size(0))
        sorted_max_activating_image_label_indices = max_activating_image_label_indices[neurons.to('cpu')]
        std_per_row = torch.std(sorted_max_activating_image_label_indices[:,:10].float(), dim=1)
        top_counts = range(100, sae_mean_acts.size(0), 100)
        average_stds = {}
        for count in top_counts:
            avg_std = std_per_row[:count].mean()
            average_stds[count] = avg_std.item()
            print(f'{count}: {avg_std.item()}')
        with open(f'{directory}/average_stds.json', 'w') as f:
            json.dump(average_stds, f)
            
    save_highest_activating_images_high(max_activating_image_indices[:,:10], max_activating_image_values[:,:10], directory, dataset, image_key, sae_mean_acts, sae_sparsity, max_activating_image_label_indices)
