import os, tempfile
import sys
import torch
sys.path.append("..")
from training.config import SDSAERunnerConfig
from training.save_feature import save_features
from training.sd_activations_store import SDActivationsStore
import wandb
import re
from training.utils import SDkSAEloader
from torch.optim import Adam
from tqdm import tqdm
import torch.nn as nn
import wandb
from training.optim import get_scheduler
from training.k_sparse_autoencoder import KSparseAutoencoder
import argparse

def train_ksae_on_sd(
    k_sparse_autoencoder: KSparseAutoencoder,
    activation_store: SDActivationsStore,
):
    batch_size = k_sparse_autoencoder.cfg.batch_size
    total_training_tokens = k_sparse_autoencoder.cfg.total_training_tokens
    
    if k_sparse_autoencoder.cfg.log_to_wandb:
        wandb.init(project="Revelio")

    total_training_steps = total_training_tokens // batch_size
    n_training_steps = 0
    n_training_tokens = 0

    # track active features
    act_freq_scores = torch.zeros(k_sparse_autoencoder.cfg.d_sae, device=k_sparse_autoencoder.cfg.device)
    n_forward_passes_since_fired = torch.zeros(k_sparse_autoencoder.cfg.d_sae, device=k_sparse_autoencoder.cfg.device)
    n_frac_active_tokens = 0
    
    optimizer = Adam(k_sparse_autoencoder.parameters(),
                     lr = k_sparse_autoencoder.cfg.lr)
    scheduler = get_scheduler(
        k_sparse_autoencoder.cfg.lr_scheduler_name,
        optimizer=optimizer,
        warm_up_steps = k_sparse_autoencoder.cfg.lr_warm_up_steps, 
        training_steps=total_training_steps,
        lr_end=k_sparse_autoencoder.cfg.lr / 10, 
    )
    k_sparse_autoencoder.initialize_b_dec(activation_store)
    k_sparse_autoencoder.train()
    

    pbar = tqdm(total=total_training_tokens, desc="Training SAE")
    while n_training_tokens < total_training_tokens:

        k_sparse_autoencoder.set_decoder_norm_to_unit_norm()
            
        scheduler.step()
        optimizer.zero_grad()
        
        sae_in = activation_store.next_batch().to(k_sparse_autoencoder.cfg.device)
        
        sae_out, feature_acts, loss = k_sparse_autoencoder(
            sae_in,
        )
        did_fire = ((feature_acts > 0).float().sum(-2) > 0)
        n_forward_passes_since_fired += 1
        n_forward_passes_since_fired[did_fire] = 0
        
        n_training_tokens += batch_size

        with torch.no_grad():
            act_freq_scores += (feature_acts.abs() > 0).float().sum(0)
            n_frac_active_tokens += batch_size
            feature_sparsity = act_freq_scores / n_frac_active_tokens

            if k_sparse_autoencoder.cfg.log_to_wandb and ((n_training_steps + 1) % k_sparse_autoencoder.cfg.wandb_log_frequency == 0):
                # metrics for currents acts
                l0 = (feature_acts > 0).float().sum(-1).mean()
                current_learning_rate = optimizer.param_groups[0]["lr"]
                
                per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).squeeze()
                total_variance = sae_in.pow(2).sum(-1)
                explained_variance = 1 - per_token_l2_loss/total_variance
                
                wandb.log(
                    {
                        # losses
                        "losses/overall_loss": loss.item(),
                        # variance explained
                        "metrics/explained_variance": explained_variance.mean().item(),
                        "metrics/explained_variance_std": explained_variance.std().item(),
                        "metrics/l0": l0.item(),
                        # sparsity
                        "sparsity/mean_passes_since_fired": n_forward_passes_since_fired.mean().item(),
                        "sparsity/dead_features": (
                            feature_sparsity < k_sparse_autoencoder.cfg.dead_feature_threshold
                        )
                        .float()
                        .mean()
                        .item(),
                        "details/n_training_tokens": n_training_tokens,
                        "details/current_learning_rate": current_learning_rate,
                    },
                    step=n_training_steps,
                )

            if k_sparse_autoencoder.cfg.log_to_wandb and ((n_training_steps + 1) % k_sparse_autoencoder.cfg.wandb_log_frequency == 0):
                if "cuda" in str(k_sparse_autoencoder.cfg.device):
                    torch.cuda.empty_cache()

            pbar.set_description(
                f"{n_training_steps}| MSE Loss {loss.item():.3f}"
            )
            pbar.update(batch_size)

        loss.backward()
        k_sparse_autoencoder.remove_gradient_parallel_to_decoder_directions()
        optimizer.step()
        n_training_steps += 1
        
    return k_sparse_autoencoder

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


def parse_args():
    parser = argparse.ArgumentParser(description="Parse SDSAERunnerConfig parameters")

    # Add arguments with defaults
    parser.add_argument('--model_name', type=str, default='caltech101/SDv1-5/timestep_25/mid_block', help="directory for extracted features to train k-SAE")
    parser.add_argument('--feature_dir', type=str, default='features/caltech101/SDv1-5/step25_mid', help="Directory for saving features")
    parser.add_argument('--module_name', type=str, default='mid_block', help="Module name")
    parser.add_argument('--dataset_name', type=str, default='dpdl-benchmark/caltech101', help="Huggingface Dataset name")
    parser.add_argument('--use_cached_activations', action='store_true', help="Use cached activations", default=True)
    parser.add_argument('--d_in', type=int, default=1280, help="Input dimensionality")

    # SAE Parameters
    parser.add_argument('--expansion_factor', type=int, default=64, help="Expansion factor")
    parser.add_argument('--b_dec_init_method', type=str, default='mean', help="Decoder initialization method")
    parser.add_argument('--k', type=int, default=32, help="Number of clusters")

    # Training Parameters
    parser.add_argument('--lr', type=float, default=0.0004, help="Learning rate")
    parser.add_argument('--lr_scheduler_name', type=str, default='constantwithwarmup', help="Learning rate scheduler name")
    parser.add_argument('--batch_size', type=int, default=8192, help="Batch size")
    parser.add_argument('--lr_warm_up_steps', type=int, default=500, help="Number of warm-up steps")
    parser.add_argument('--total_training_tokens', type=int, default=83886080, help="Total training tokens")
    parser.add_argument('--dead_feature_threshold', type=float, default=1e-6, help="Dead feature threshold")

    # WANDB
    parser.add_argument('--log_to_wandb', action='store_true', default=True, help="Log to WANDB")
    parser.add_argument('--wandb_project', type=str, default='revelio', help="WANDB project name")
    parser.add_argument('--wandb_entity', type=str, default=None, help="WANDB entity")
    parser.add_argument('--wandb_log_frequency', type=int, default=20, help="WANDB log frequency")

    # Misc
    parser.add_argument('--device', type=str, default="cuda", help="Device to use (e.g., cuda, cpu)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--checkpoint_path', type=str, default="Checkpoints", help="Checkpoint path")
    parser.add_argument('--dtype', type=str, default="float32", help="Data type (e.g., float32)")

    return parser.parse_args()

def args_to_config(args):
    return SDSAERunnerConfig(
        model_name=args.model_name,
        layer_name=args.layer_name,
        feature_dir=args.feature_dir,
        module_name=args.module_name,
        dataset_name=args.dataset_name,
        use_cached_activations=args.use_cached_activations,
        d_in=args.d_in,
        expansion_factor=args.expansion_factor,
        b_dec_init_method=args.b_dec_init_method,
        k=args.k,
        lr=args.lr,
        lr_scheduler_name=args.lr_scheduler_name,
        batch_size=args.batch_size,
        lr_warm_up_steps=args.lr_warm_up_steps,
        total_training_tokens=args.total_training_tokens,
        dead_feature_threshold=args.dead_feature_threshold,
        log_to_wandb=args.log_to_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_log_frequency=args.wandb_log_frequency,
        device=args.device,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path,
        dtype=getattr(torch, args.dtype)
    )

if __name__ == "__main__":

    args = parse_args()
    cfg = args_to_config(args)
    print(cfg)

    torch.cuda.empty_cache()
    k_sparse_autoencoder = sd_ksae_runner(cfg)

    k_sparse_autoencoder.eval()
    activation_store = SDActivationsStore(cfg)
    save_features(
        k_sparse_autoencoder,
        activation_store,
        number_of_images = 24790,
        number_of_max_activating_images = 20,
    )

