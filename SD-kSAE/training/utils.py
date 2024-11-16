from typing import Tuple
from training.sd_activations_store import SDActivationsStore
from training.config import SDSAERunnerConfig
from training.k_sparse_autoencoder import KSparseAutoencoder
from training.hooked_sd import HookedStableDiffusion

class SDkSAEloader():

    def __init__(self, cfg: SDSAERunnerConfig):
        self.cfg = cfg
        
    def load_session(self) -> Tuple[HookedStableDiffusion, KSparseAutoencoder, SDActivationsStore]:

        activations_loader = self.get_activations_loader(self.cfg)
        k_sparse_autoencoder = self.initialize_sparse_autoencoder(self.cfg)

        return k_sparse_autoencoder, activations_loader

    
    def get_model(self, model_name: str, model_name_proc:str, image_size:str):
        
        model = HookedStableDiffusion(model_name, model_name_proc, image_size)
        model.eval()
        
        return model 

    
    def initialize_sparse_autoencoder(self, cfg: SDSAERunnerConfig):

        k_sparse_autoencoder = KSparseAutoencoder(cfg)
        
        return k_sparse_autoencoder


    def get_activations_loader(self, cfg: SDSAERunnerConfig):

        
        activations_loader = SDActivationsStore(cfg)
        
        return activations_loader
