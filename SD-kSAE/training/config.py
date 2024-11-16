from dataclasses import dataclass
from typing import Optional
import torch
import wandb

@dataclass
class SDSAERunnerConfig():

    image_size: int = 512,
    num_sampling_steps: int = 25,
    vae: str = "mse"

    model_name: str = "openai/clip-vit-base-patch32"
    model_name_proc: str= "openai/clip-vit-base-patch32"
    timestep: int = 0
    module_name: str = "mid_block"
    feature_dir: str = None
    layer_name:str = None
    block_layer: int = 10
    dataset_name: str = "evanarlian/imagenet_1k_resized_256"
    use_cached_activations: bool = False
    block_name :str = 'mid_block'
    image_key: str = 'image'

    # SAE Parameters
    d_in: int = 768
    k: int = 32

    # Activation Store Parameters
    total_training_tokens: int = 2_000_000
    
    # Misc
    device: str = "cpu"
    seed: int = 42
    dtype: torch.dtype = torch.float32

    # SAE Parameters
    b_dec_init_method: str = "mean"
    expansion_factor: int = 4
    from_pretrained_path: Optional[str] = None

    # Training Parameters
    lr: float = 3e-4
    lr_scheduler_name: str = "constant"  
    lr_warm_up_steps: int = 500
    batch_size: int = 4096
    sae_batch_size: int = 1024,
    dead_feature_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "revelio"
    wandb_entity: str = None
    wandb_log_frequency: int = 10

    # Misc
    checkpoint_path: str = "checkpoints"
    max_batch_size: int = 32
    save_path: str = 'feature'
    def __post_init__(self):
        
        self.d_sae = self.d_in * self.expansion_factor

        self.run_name = f"{self.d_sae}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"

        if self.b_dec_init_method not in ["mean"]:
            raise ValueError(
                f"b_dec_init_method must be geometric_median, mean, or zeros. Got {self.b_dec_init_method}"
            )

        self.device = torch.device(self.device)

        unique_id = wandb.util.generate_id()
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}"

        print(
            f"Run name: {self.d_sae}-LR-{self.lr}-Tokens-{self.total_training_tokens:3.3e}"
        )
        # Print out some useful info:

        total_training_steps = self.total_training_tokens // self.batch_size
        print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        print(f"Total wandb updates: {total_wandb_updates}")
        
