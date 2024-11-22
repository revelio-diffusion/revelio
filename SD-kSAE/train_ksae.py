import os
import sys
import torch
sys.path.append("..")
from training.config import SDSAERunnerConfig
from training.sd_runner import sd_ksae_runner
from training.save_feature import get_feature_data
from training.sd_activations_store import SDActivationsStore

cfg = SDSAERunnerConfig(
    model_name = 'caltech101/SD15/timestep_500',    
    layer_name = "mid",
    feature_dir = 'feature/SD15/Caltech101/step500_mid',
    module_name = "mid_block",
    dataset_path = "dpdl-benchmark/caltech101",
    use_cached_activations = True,
    d_in = 1280,

    # SAE Parameters
    expansion_factor = 64,
    b_dec_init_method = "mean",
    k= 32,

    # Training Parameters
    lr = 0.0004,
    lr_scheduler_name="constantwithwarmup",
    batch_size = 8192,
    lr_warm_up_steps=500,
    total_training_tokens = 83_886_080,
    dead_feature_threshold = 1e-6,
    
    # WANDB
    log_to_wandb = True,
    wandb_project = "revelio",
    wandb_entity = None,
    wandb_log_frequency=20,
    
    # Misc
    device = "cuda",
    seed = 42,
    checkpoint_path = "Checkpoints",
    dtype = torch.float32,

    )

torch.cuda.empty_cache()
k_sparse_autoencoder = sd_ksae_runner(cfg)
k_sparse_autoencoder.eval()

activation_store = SDActivationsStore(cfg)

get_feature_data(
    k_sparse_autoencoder,
    activation_store,
    number_of_images = 24790,
    number_of_max_activating_images = 20,
)

print("*****Done*****")
