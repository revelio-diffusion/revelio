import torch
import torch.nn as nn
import numpy as np

# from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
# from transformers import AutoImageProcessor, AutoModel, AutoConfig
from typing import Optional, Union
from torchvision import transforms
from diffusers import DDIMScheduler, StableDiffusionPipeline
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel


class ProcessorWrapper:
    def __init__(
        self,
        transform,
        height=378,
        width=378,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
    ):
        self._crop_size = {
            "height": height,
            "width": width,
        }
        self._transforms = transform
        self.image_mean = image_mean

    @property
    def crop_size(self):
        return self._crop_size

    def preprocess(self, image, return_tensors="pt"):
        output = {}
        output["pixel_values"] = [self._transforms(image)]
        return output


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


class DiffusionVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")
        # self._config = args
        # print(self._config)
        # exit()

        # self.load_model()

        if not delay_load:
            self.load_model()
        # else:
        #     self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        self.vision_model = "diffusion"
        sd_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"

        # Load the UNet model and pipeline
        unet = MyUNet2DConditionModel.from_pretrained(sd_id, subfolder="unet")
        onestep_pipe = OneStepSDPipeline.from_pretrained(
            sd_id, unet=unet, safety_checker=None, low_cpu_mem_usage=False
        )
        onestep_pipe.vae.decoder = None
        onestep_pipe.scheduler = DDIMScheduler.from_pretrained(
            sd_id, subfolder="scheduler"
        )

        # Exclude diffusion_pipe from nn.Module's parameters
        object.__setattr__(self, "diffusion_pipe", onestep_pipe)

        # Freeze all diffusion components
        self.diffusion_pipe.vae.requires_grad_(False)
        self.diffusion_pipe.unet.requires_grad_(False)
        self.diffusion_pipe.text_encoder.requires_grad_(False)

        # Determine the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Move diffusion components to the desired device
        self.diffusion_pipe.to(self.device)

        # Additional components as required
        self.vae = self.diffusion_pipe.vae
        self.unet = self.diffusion_pipe.unet
        self.scheduler = self.diffusion_pipe.scheduler
        self.diffusion_pipe.output_tokens = True

        # Load precomputed empty_prompt_embeds
        self.empty_prompt_embeds = torch.load("empty_prompt_embeds.pt")
        self.empty_prompt_embeds = self.empty_prompt_embeds.to(
            device=self.device, dtype=self.dtype
        )

        # Define model configurations and processor
        self._hidden_size = 3520
        self._image_size = 512
        self._patch_size = 16
        self.up_ft_index = [0, 1, 2, 3]

        preprocess = transforms.Compose(
            [
                transforms.Resize(self._image_size),
                transforms.CenterCrop(self._image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.image_processor = ProcessorWrapper(
            preprocess,
            height=self._image_size,
            width=self._image_size,
            image_mean=[0.5, 0.5, 0.5],
        )

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # image_features = image_forward_outs["x_prenorm"]
        # if self.select_feature == 'patch':
        #     image_features = image_features[:, 1:]
        # elif self.select_feature == 'cls_patch':
        #     image_features = image_features
        # else:
        #     raise ValueError(f'Unexpected select feature: {self.select_feature}')
        image_features = image_forward_outs["up_ft"][1]
        return image_features

    def extract_raw_features(self, images, prompt_embeds, time_step):
        batch_size = images.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self.empty_prompt_embeds.repeat(batch_size, 1, 1).to(
                device=self.device, dtype=self.dtype
            )

        with torch.no_grad():
            images = images.to(self.device)
            scale_factor = self.vae.config.scaling_factor
            latents = scale_factor * self.vae.encode(images).latent_dist.mode()

            t = torch.tensor(time_step, dtype=torch.long, device=self.device)
            noise = torch.randn_like(latents, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents, noise, t).to(
                dtype=self.dtype, device=self.device
            )

            unet_output = self.unet(
                latents_noisy,
                t,
                up_ft_indices=self.up_ft_index,
                encoder_hidden_states=prompt_embeds.detach(),
            )

        return unet_output

    # @torch.no_grad()
    # def forward(self, images):
    #     if type(images) is list:
    #         image_features = []
    #         for image in images:
    #             image_forward_out = self.vision_tower.forward_features(image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
    #             image_feature = self.feature_select(image_forward_out).to(image.dtype)
    #             image_features.append(image_feature)
    #     else:
    #         image_forward_outs = self.vision_tower.forward_features(images.to(device=self.device, dtype=self.dtype))
    #         image_features = self.feature_select(image_forward_outs).to(images.dtype)

    #     return image_features
    @torch.no_grad()
    def forward(self, images):
        images = images.to(self.device, dtype=self.dtype)
        image_forward_outs = self.extract_raw_features(images, None, 25)
        # image_forward_outs = self.extract_raw_features(images, None, 200)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        # print(f"[{torch.distributed.get_rank()}] Selected features")
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.diffusion_pipe.vae.dtype

    @property
    def model_device(self):
        return self.diffusion_pipe.vae.device

    @property
    def config(self):
        if self.is_loaded:
            return self.clip_vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # return self.config.hidden_size
        return 1024

    @property
    def num_patches(self):
        # return (self.config.image_size // self.config.patch_size) ** 2
        return 256
