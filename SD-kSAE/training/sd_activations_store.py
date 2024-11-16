import os, re
import torch
from torch.utils.data import DataLoader, Dataset
from training.config import SDSAERunnerConfig

class CustomFeatureDataset(Dataset):
    def __init__(self, dataset_dir, layer_dir):
        """
        Custom dataset that preloads features and labels from .npy files into memory.
        
        Args:
            feature_dir (str): Path to the directory containing feature .npy files.
            label_dir (str): Path to the directory containing label .npy files.
        """
        self.activations = []
        self.feature_dir = os.path.join(dataset_dir, layer_dir)
        
        # # Load the cached activations and labels
        activation_files = sorted(
            [f for f in os.listdir(self.feature_dir) if f.endswith('.pt')],
            key=lambda x: tuple(map(int, re.search(r'image_(\d+)_(\d+)_features.pt', x).groups())) if re.search(r'image_(\d+)_(\d+)_features.pt', x) else (float('inf'), float('inf'))  # Assign inf if no match
        )
        for activation_file in activation_files:
            self.activations.append(torch.load(os.path.join(self.feature_dir, activation_file), weights_only=True).mean((-1,-2)))
            # self.activations.append(torch.load(os.path.join(self.feature_dir, activation_file), weights_only=True).mean(1))

        self.activations = torch.cat(self.activations, dim=0)
    
    def __len__(self):
        """
        Return the total number of samples (features/labels).
        """
        return len(self.activations)
    
    def __getitem__(self, idx):
        """
        Get the feature and label for a given index from memory.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            Tuple[Tensor, Tensor]: A tuple containing the feature tensor and label tensor.
        """
        # Retrieve preloaded feature and label
        feature = self.activations[idx].clone().detach()
        
        return feature

class SDActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs. 
    """
    def __init__(
        self, cfg: SDSAERunnerConfig, create_dataloader: bool = True, train=True,
    ):
        self.cfg = cfg
        if self.cfg.use_cached_activations:
            self.feature_dataset = CustomFeatureDataset(self.cfg.model_name, self.cfg.layer_name)
            self.feature_loader = DataLoader(self.feature_dataset, batch_size=self.cfg.batch_size, shuffle=True)
            self.loader_iter = iter(self.feature_loader)
    
    def next_batch(self):
        """
        Retrieve the next batch of activations [batch_size, dim].
        
        Returns:
            Tensor: A tensor containing the activations for the batch in the shape [batch_size, dim].
        """
        try:
            # Get the next batch from the DataLoader
            activations = next(self.loader_iter)
        except StopIteration:
            # If the DataLoader is exhausted, reinitialize the iterator
            self.loader_iter = iter(self.feature_loader)
            activations = next(self.loader_iter)
        
        return activations

