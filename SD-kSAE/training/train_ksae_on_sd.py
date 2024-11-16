import torch
from torch.optim import Adam
from tqdm import tqdm
import wandb
from training.sd_activations_store import SDActivationsStore
from training.optim import get_scheduler
from training.k_sparse_autoencoder import KSparseAutoencoder

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
