import wandb
import re
from training.train_ksae_on_sd import train_ksae_on_sd
from training.utils import SDkSAEloader

def sd_ksae_runner(cfg):
    loader = SDkSAEloader(cfg)
    k_sparse_autoencoder, activations_loader = loader.load_session()

    if cfg.log_to_wandb:
        wandb.init(project=cfg.wandb_project, config=cfg, name=cfg.run_name)
    
    # train SAE
    k_sparse_autoencoder = train_ksae_on_sd(
        k_sparse_autoencoder, activations_loader,
    )

    # save sae to checkpoints folder
    path = f"{cfg.checkpoint_path}/final_{k_sparse_autoencoder.get_name()}.pt"
    k_sparse_autoencoder.save_model(path)
    
    # upload to wandb
    if cfg.log_to_wandb:
        model_artifact = wandb.Artifact(
            f"{re.sub(r'[^a-zA-Z0-9]', '', k_sparse_autoencoder.get_name())}", type="model", metadata=dict(cfg.__dict__)
        )
        model_artifact.add_file(path)
        wandb.log_artifact(model_artifact, aliases=["final_model"])
        

    if cfg.log_to_wandb:
        wandb.finish()
        
    return k_sparse_autoencoder