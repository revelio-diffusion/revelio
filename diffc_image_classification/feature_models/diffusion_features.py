import torch
import numpy as np
from .base_encoder import ProcessorWrapper
from .base_encoder import BaseVisionTower
from typing import Optional, Union, Dict, Any
from torchvision import transforms
from diffusers import DDIMScheduler, StableDiffusionPipeline, DiTPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.models.transformers.dit_transformer_2d import DiTTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput
import torch.nn.functional as F
from diffusers.models.attention import BasicTransformerBlock
from diffusers.models.modeling_utils import ModelMixin


class MyUNet2DConditionModel(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        up_ft_indices=None,
        down_ft_indices=None,  # Add down_ft_indices to specify layers to capture downsample features
        encoder_hidden_states: torch.Tensor = None,
        return_bottleneck=False,
    ):
        r"""
        Args:
            sample (`torch.FloatTensor`):
                (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`):
                (batch, sequence_length, feature_dim) encoder hidden states
            return_bottleneck (`bool`): whether to return bottleneck (h-space) features.
        """
        default_overall_up_factor = 2**self.num_upsamplers
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            forward_upsample_size = True

        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        timesteps = timestep
        if len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps).to(dtype=self.dtype)
        emb = self.time_embedding(t_emb, None)

        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        down_ft = {}  # Dictionary to store downsampling features

        for i, downsample_block in enumerate(self.down_blocks):
            _has_attr = hasattr(downsample_block, "has_cross_attention")
            if _has_attr and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=None,
                    cross_attention_kwargs=None,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

            down_block_res_samples += res_samples

            # # Capture downsampling features if specified
            # if down_ft_indices is not None and i in down_ft_indices:
            down_ft[i] = sample

        # Mid block processing
        sample = self.mid_block(
            sample,
            emb,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=None,
            cross_attention_kwargs=None,
        )

        # Extract bottleneck (h-space) features if required
        bottleneck_features = {0: sample}

        up_ft = {}
        if up_ft_indices is not None:
            for i, upsample_block in enumerate(self.up_blocks):
                if i > np.max(up_ft_indices):
                    break

                is_final_block = i == len(self.up_blocks) - 1

                res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
                down_block_res_samples = down_block_res_samples[
                    : -len(upsample_block.resnets)
                ]

                if not is_final_block and forward_upsample_size:
                    upsample_size = down_block_res_samples[-1].shape[2:]

                _has_attr = hasattr(upsample_block, "has_cross_attention")
                if _has_attr and upsample_block.has_cross_attention:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        encoder_hidden_states=encoder_hidden_states,
                        cross_attention_kwargs=None,
                        upsample_size=upsample_size,
                        attention_mask=None,
                    )
                else:
                    sample = upsample_block(
                        hidden_states=sample,
                        temb=emb,
                        res_hidden_states_tuple=res_samples,
                        upsample_size=upsample_size,
                    )

                if i in up_ft_indices:
                    up_ft[i] = sample

        sample = self.conv_out(sample)

        output = {
            "up_ft": up_ft,
            "down_ft": down_ft,  # Include downsampling features in the output
            "bottleneck": bottleneck_features,
            "sample": sample,
        }

        return output


class MyDiTTransformer2DModel(DiTTransformer2DModel):
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        return_dict: bool = True,
    ):
        """
        The [`DiTTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, channel, height, width)` if continuous):
                Input `hidden_states`.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        hidden_states_list = []

        # 1. Input
        height, width = (
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        hidden_states = self.pos_embed(hidden_states)

        # hidden_states_list.append(hidden_states)

        # 2. Blocks
        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                )

            hidden_states_list.append(hidden_states)

        hidden_states_cat = torch.stack(hidden_states_list, dim=0)
        hidden_states_cat = hidden_states_cat.permute(
            1, 0, 2, 3
        )  # (batch, num_layers, tokens, hidden_dim)
        return hidden_states_cat


class OneStepSDPipeline(StableDiffusionPipeline):
    def __call__(
        self,
        img_tensor,
        t,
        up_ft_indices=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        return_bottleneck=False,
    ):
        device = self._execution_device

        scale_factor = self.vae.config.scaling_factor
        latents = scale_factor * self.vae.encode(img_tensor).latent_dist.mode()

        t = torch.tensor(t, dtype=torch.long, device=device)
        noise = torch.randn_like(latents).to(device)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        unet_output = self.unet(
            latents_noisy,
            t,
            up_ft_indices=up_ft_indices,
            encoder_hidden_states=prompt_embeds,
            return_bottleneck=return_bottleneck,
        )
        return unet_output


class DiffusionVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, config, delay_load=False):
        super(DiffusionVisionTower, self).__init__(
            vision_tower_name, config, delay_load
        )
        self._config = config

        if not self.delay_load:
            self.load_model()

    def extract_bottleneck_features(
        self, images, time_step, output="dense", layers=[1, 2, 3]
    ):
        batch_size = images.shape[0]

        prompt_embeds = self.empty_prompt_embeds.repeat(batch_size, 1, 1).to(
            device=self.device, dtype=self.dtype
        )

        with torch.no_grad():
            scale_factor = self.vae.config.scaling_factor
            latents = scale_factor * self.vae.encode(images).latent_dist.mode()

            t = torch.tensor(time_step, dtype=torch.long, device=self.device)
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t).to(
                dtype=self.dtype, device=self.device
            )

            unet_output = self.unet(
                latents_noisy,
                t,
                up_ft_indices=self.up_ft_index,
                encoder_hidden_states=prompt_embeds.detach(),
                return_bottleneck=True,
            )

        # Extract the bottleneck features
        bottleneck_features = unet_output["bottleneck"]

        return bottleneck_features

    def extract_raw_features(self, images, prompt_embeds, time_step):

        batch_size = images.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self.empty_prompt_embeds.repeat(batch_size, 1, 1).to(
                device=self.device, dtype=self.dtype
            )

        if self._config["diffusion_step_type"] == "onestep":

            with torch.no_grad():
                scale_factor = self.vae.config.scaling_factor
                latents = scale_factor * self.vae.encode(images).latent_dist.mode()

                t = torch.tensor(time_step, dtype=torch.long, device=self.device)
                noise = torch.randn_like(latents)
                latents_noisy = self.scheduler.add_noise(latents, noise, t).to(
                    dtype=self.dtype, device=self.device
                )

                unet_output = self.unet(
                    latents_noisy,
                    t,
                    up_ft_indices=self.up_ft_index,
                    encoder_hidden_states=prompt_embeds.detach(),
                )

        else:
            raise NotImplementedError
        return unet_output

    def load_model(self, device_map=None):
        self.vision_model = "diffusion"
        sd_id = self._config["model_name"]
        # sd_id = "stabilityai/stable-diffusion-2-1"

        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")

        if self._config["diffusion_step_type"] == "onestep":
            onestep_pipe = OneStepSDPipeline.from_pretrained(
                sd_id, unet=unet, safety_checker=None
            )
            onestep_pipe.vae.decoder = None
            onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
                sd_id, subfolder="scheduler"
            )
            diffusion_pipe = onestep_pipe

        else:
            raise NotImplementedError

        self.diffusion_pipe = diffusion_pipe

        self.vae = self.diffusion_pipe.vae
        self.unet = self.diffusion_pipe.unet
        self.scheduler = self.diffusion_pipe.scheduler

        self.up_ft_index = self._config.get("up_ft_index", [0, 1, 2, 3])
        self.diffusion_pipe.output_tokens = True

        with torch.no_grad():
            self.empty_prompt_embeds = self.diffusion_pipe.encode_prompt(
                [""],
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            self.empty_prompt_embeds = self.empty_prompt_embeds[0]

        self._hidden_size = self._config.get("hidden_size", 3520)
        self._image_size = self._config.get("image_size", 512)
        self._patch_size = self._config.get("patch_size", 16)

        preprocess = transforms.Compose(
            [
                transforms.Resize(self._image_size),
                transforms.CenterCrop(self._image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.image_processor = ProcessorWrapper(
            preprocess, height=self._image_size, width=self._image_size
        )

        self.unet.requires_grad_(self._config.get("unfreeze_mm_vision_tower", False))
        self.is_loaded = True

    def _forward(self, images):
        with torch.set_grad_enabled(
            self._config.get("unfreeze_mm_vision_tower", False)
        ):
            image_features = self.extract_features(
                images.to(device=self.device, dtype=self.dtype)
            ).to(images.dtype)
            return image_features

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def image_size(self):
        return self._image_size

    @property
    def image_token_len(self):
        return (self.image_size // self.patch_size) ** 2

    @property
    def dtype(self):
        if hasattr(self.vae, "dtype"):
            return self.vae.dtype
        else:
            params = list(self.vae.parameters())
            return params[0].dtype if len(params) > 0 else torch.float32

    @property
    def device(self):
        if hasattr(self.vae, "device"):
            return self.vae.device
        else:
            params = list(self.vae.parameters())
            return params[0].device if len(params) > 0 else torch.device("cpu")


class OneStepDiTPipeline(DiTPipeline):
    def __call__(
        self,
        img_tensor,
        t,
        class_labels=None,
        up_ft_indices=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        return_bottleneck=False,
    ):
        device = self._execution_device

        scale_factor = self.vae.config.scaling_factor
        latents = scale_factor * self.vae.encode(img_tensor).latent_dist.mode()

        t = torch.tensor(t, dtype=torch.long, device=device)

        dit_output = self.transformer(latents, t, class_labels=class_labels)
        return dit_output


class DiffusionFeatures:
    def __init__(self, config):
        self._config = config
        self.device = torch.device("cpu")  # Placeholder; will be set by accelerator
        self._init_feature_encoder()

    def _init_feature_encoder(self):
        vision_tower_name = self._config["model_name"]
        self.model = DiffusionVisionTower(
            vision_tower_name, self._config, delay_load=False
        )
        self.model.eval()
        self.preprocess = self.model.image_processor

    def get_raw_features(self, batch_images, prompt_embeds, time_step):
        return self.model.extract_raw_features(batch_images, prompt_embeds, time_step)


class DiTTower(BaseVisionTower):
    def __init__(self, vision_tower_name, config, delay_load=False):
        super(DiTTower, self).__init__(vision_tower_name, config, delay_load)
        self._config = config

        if not self.delay_load:
            self.load_model()

    def extract_raw_features(
        self, images, prompt_embeds, time_step, layer="bottleneck"
    ):
        batch_size = images.shape[0]

        # if prompt_embeds is None:
        #     prompt_embeds = self.empty_prompt_embeds.repeat(batch_size, 1, 1).to(
        #         device=self.device, dtype=self.dtype
        #     )

        # images = images.to(device="cuda")

        time_step = torch.tensor([time_step], dtype=torch.long, device=images.device)
        class_labels = torch.tensor([1000], device=images.device).repeat(batch_size)

        # print(images.shape, class_labels.shape)
        # exit()

        with torch.no_grad():
            image_features = self.model(images, t=time_step, class_labels=class_labels)

        return image_features

    def load_model(self, device_map=None):
        self.vision_model = "dit"
        sd_id = self._config["model_name"]

        onestep_pipe = OneStepDiTPipeline.from_pretrained(sd_id)
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )
        self.dit_pipe = onestep_pipe

        self.dit_pipe.transformer = MyDiTTransformer2DModel.from_pretrained(
            sd_id, subfolder="transformer"
        )
        self.transformer = self.dit_pipe.transformer
        self.vae = self.dit_pipe.vae
        self.scheduler = self.dit_pipe.scheduler
        self.model = self.dit_pipe

        self._image_size = 512
        preprocess = transforms.Compose(
            [
                transforms.Resize(self._image_size),
                transforms.CenterCrop(self._image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.image_processor = ProcessorWrapper(
            preprocess, height=self._image_size, width=self._image_size
        )

        self.is_loaded = True
        # self.device = self.vae.device

    def _forward(self, images):
        with torch.set_grad_enabled(
            self._config.get("unfreeze_mm_vision_tower", False)
        ):
            image_features = self.extract_features(
                images.to(device=self.device, dtype=self.dtype)
            ).to(images.dtype)
            return image_features


class DiffusionFeaturesDiT:
    def __init__(self, config):
        self._config = config
        self.device = torch.device("cpu")  # Placeholder; will be set by accelerator
        self._init_feature_encoder()

    def _init_feature_encoder(self):
        vision_tower_name = self._config["model_name"]
        self.model = DiTTower(vision_tower_name, self._config, delay_load=False)
        self.model.eval()
        self.preprocess = self.model.image_processor

    def get_raw_features(self, batch_images, prompt_embeds, time_step):
        return self.model.extract_raw_features(batch_images, prompt_embeds, time_step)
