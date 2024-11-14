import sys
import torch
from training.feature_extractor import get_feature_data
sys.path.append("..")
from training.config import SDSAERunnerConfig
from training.utils import SDkSAEloader
    
cfg = SDSAERunnerConfig(
    image_size = 256,
    model_name_proc = "openai/clip-vit-large-patch14",
    model_name = "runwayml/stable-diffusion-v1-5",
    timestep = 500,
    module_name = "mid_block",    
    dataset_path = "dpdl-benchmark/caltech101",
    max_batch_size = 32,

    # Misc
    device = "cuda",
    seed = 42,
    checkpoint_path = "checkpoints",
    dtype = torch.float32,
    )


loader = SDkSAEloader(cfg)
model = loader.get_model(cfg.model_name, cfg.model_name_proc, cfg.image_size)

model.to(cfg.device)
get_feature_data(
    cfg,
    model
)