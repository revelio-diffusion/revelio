import gzip
import os
import pickle
import einops
import torch
from torch import Tensor, nn
from transformer_lens.hook_points import HookedRootModule, HookPoint
from training.sd_activations_store import SDActivationsStore
from training.config import SDSAERunnerConfig
from typing import NamedTuple

class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""

class KSparseAutoencoder(HookedRootModule):

    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        if not isinstance(self.d_in, int):
            raise ValueError(
                f"d_in must be an int but was {self.d_in=}; {type(self.d_in)=}"
            )
        self.d_sae = cfg.d_sae
        self.dtype = cfg.dtype
        self.device = cfg.device

        # NOTE: if using resampling neurons method, you must ensure that we initialise the weights in the order W_enc, b_enc, W_dec, b_dec
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_in, self.d_sae, dtype=self.dtype, device=self.device)
            )   
        )
        self.b_enc = nn.Parameter(
            torch.zeros(self.d_sae, dtype=self.dtype, device=self.device)
        )

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_sae, self.d_in, dtype=self.dtype, device=self.device)
            )
        )

        with torch.no_grad():
            self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)

        self.b_dec = nn.Parameter(
            torch.zeros(self.d_in, dtype=self.dtype, device=self.device)
        )

        self.hook_sae_in = HookPoint()
        self.hook_hidden_pre = HookPoint()
        self.hook_hidden_post = HookPoint()
        self.hook_sae_out = HookPoint()
        self.setup()  

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def decoder_impl(self, top_indices: Tensor, top_acts: Tensor, W_dec: Tensor):
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (W_dec.shape[-1],))
        acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)
        return acts @ W_dec.mT

    def forward(self, x):
        x = x.to(self.dtype)
        sae_in = self.hook_sae_in(
            x - self.b_dec
        )  

        hidden_pre = self.hook_hidden_pre(
            einops.einsum(
                sae_in,
                self.W_enc,
                "... d_in, d_in d_sae -> ... d_sae",
            )
            + self.b_enc
        )
        hidden_pre = self.hook_hidden_post(hidden_pre)
        top_acts, top_indices = self.select_topk(hidden_pre)
        sae_out = self.hook_sae_out(self.decoder_impl(top_indices, top_acts.to(self.dtype), self.W_dec.mT) + self.b_dec)
        
        mean_activations = x.mean(dim=0)  # averaging over the batch dimension
        baseline_mse = (x - mean_activations).pow(2).mean()

        actual_mse = (sae_out - x).pow(2).mean()
        mse_loss = (actual_mse / baseline_mse).nan_to_num(0)

        loss = mse_loss 

        return sae_out, hidden_pre, loss

    @torch.no_grad()
    def initialize_b_dec(self, activation_store):
        if self.cfg.b_dec_init_method == "mean":
            self.initialize_b_dec_with_mean(activation_store)
        else:
            raise ValueError(f"Unexpected b_dec_init_method: {self.cfg.b_dec_init_method}")

    @torch.no_grad()
    def initialize_b_dec_with_mean(self, activation_store):
        
        previous_b_dec = self.b_dec.clone().cpu()
        if isinstance(activation_store, SDActivationsStore):
            all_activations = activation_store.next_batch().detach().cpu()
        else:
            all_activations = activation_store.storage_buffer.detach().cpu()
            
        out = all_activations.mean(dim=0)
        
        previous_distances = torch.norm(all_activations - previous_b_dec, dim=-1)
        distances = torch.norm(all_activations - out, dim=-1)
        
        print("Reinitializing b_dec with mean of activations")
        print(f"Previous distances: {previous_distances.median(0).values.mean().item()}")
        print(f"New distances: {distances.median(0).values.mean().item()}")
        
        self.b_dec.data = out.to(self.dtype).to(self.device)

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        self.W_dec.data /= torch.norm(self.W_dec.data, dim=1, keepdim=True)
        
    @torch.no_grad()
    def remove_gradient_parallel_to_decoder_directions(self):

        parallel_component = einops.einsum(
            self.W_dec.grad,
            self.W_dec.data,
            "d_sae d_in, d_sae d_in -> d_sae",
        )
        
        self.W_dec.grad -= einops.einsum(
            parallel_component,
            self.W_dec.data,
            "d_sae, d_sae d_in -> d_sae d_in",
        )
    
    def save_model(self, path: str):

        folder = os.path.dirname(path)
        os.makedirs(folder, exist_ok=True)
        
        state_dict = {
            "cfg": self.cfg,
            "state_dict": self.state_dict()
        }
        
        if path.endswith(".pt"):
            torch.save(state_dict, path)
        elif path.endswith("pkl.gz"):
            with gzip.open(path, "wb") as f:
                pickle.dump(state_dict, f)
        else:
            raise ValueError(f"Unexpected file extension: {path}, supported extensions are .pt and .pkl.gz")
        
        
        print(f"Saved model to {path}")
    

    def get_name(self):
        if isinstance(self.cfg, SDSAERunnerConfig):
            sae_name = f"k_sparse_autoencoder_{self.cfg.model_name}_{self.cfg.block_layer}_{self.cfg.module_name}_{self.cfg.d_sae}"
        else:
            sae_name = f"k_sparse_autoencoder_{self.cfg.model_name}_{self.cfg.hook_point}_{self.cfg.d_sae}"
        return sae_name