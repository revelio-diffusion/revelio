#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_vision_projector_diffusion

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)

        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, load_model = "clip", delay_load=True)
            # self.dino_tower = build_vision_tower(config, load_model = "dino", delay_load=True)
            self.diffusion_tower = build_vision_tower(config, load_model = "diffusion", delay_load=True)

            self.mm_projector = build_vision_projector(config)
            # self.dino_mm_projector = build_vision_projector(config)
            self.diffusion_mm_projector = build_vision_projector_diffusion(config)



    def get_vision_tower(self, load_model = "clip"):
        if load_model == "clip":
            vision_tower = getattr(self, 'vision_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]

            return vision_tower
        # elif load_model == "dino":
        #     vision_tower = getattr(self, 'dino_tower', None)
        #     if type(vision_tower) is list:
        #         vision_tower = vision_tower[0]

        #     return vision_tower
        elif load_model == "diffusion":
            vision_tower = getattr(self, 'diffusion_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]

            return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        # pretrain_dino_mm_mlp_adapter = model_args.pretrain_dino_mm_mlp_adapter
        pretrain_diffusion_mm_mlp_adapter = model_args.pretrain_diffusion_mm_mlp_adapter


        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args, load_model="clip")


            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()


        # if self.get_vision_tower(load_model = "dino") is None:
        #     dino_tower = build_vision_tower(model_args, load_model="dino")


        #     if fsdp is not None and len(fsdp) > 0:
        #         self.dino_tower = [dino_tower]
        #     else:
        #         self.dino_tower = dino_tower
        # else:
        #     if fsdp is not None and len(fsdp) > 0:
        #         dino_tower = self.dino_tower[0]
        #     else:
        #         dino_tower = self.dino_tower
        #     dino_tower.load_model()

        if self.get_vision_tower(load_model = "diffusion") is None:
            diffusion_tower = build_vision_tower(model_args, load_model="diffusion")


            if fsdp is not None and len(fsdp) > 0:
                self.diffusion_tower = [diffusion_tower]
            else:
                self.diffusion_tower = diffusion_tower

        else:
            if fsdp is not None and len(fsdp) > 0:
                diffusion_tower = self.diffusion_tower[0]
            else:
                diffusion_tower = self.diffusion_tower
            diffusion_tower.load_model()

        # if self.get_vision_tower(load_model="diffusion") is None:
        #     diffusion_tower = build_vision_tower(model_args, load_model="diffusion")

        #     # Exclude diffusion_tower from being a submodule by using object.__setattr__
        #     object.__setattr__(self, 'diffusion_tower', diffusion_tower)

        # else:
        #     diffusion_tower = self.diffusion_tower
        #     diffusion_tower.load_model()

        # print(diffusion_tower)
        # exit()


        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

        # if getattr(self, 'dino_mm_projector', None) is None:
        #     self.dino_mm_projector = build_vision_projector(self.config)

        if getattr(self, 'diffusion_mm_projector', None) is None:
            self.diffusion_mm_projector = build_vision_projector_diffusion(self.config)
            # print(self.diffusion_mm_projector)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            print("Loading pretrained MM adapter!!!")
            print(mm_projector_weights.keys())

            # keys in the file

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'model.mm_projector'))

        # if pretrain_dino_mm_mlp_adapter is not None:

        #     print("Loading pretrained DINO adpater!!!")
        #     dino_mm_projector_weights = torch.load(pretrain_dino_mm_mlp_adapter, map_location='cpu')
        #     def get_w(weights, keyword):
        #         return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

        #     self.dino_mm_projector.load_state_dict(get_w(dino_mm_projector_weights, 'dino_mm_projector'))

        if pretrain_diffusion_mm_mlp_adapter is not None:

            print("Loading pretrained Diffusion adapter!!!")
            diffusion_mm_projector_weights = torch.load(pretrain_diffusion_mm_mlp_adapter, map_location='cpu')

            print(diffusion_mm_projector_weights.keys())
            print("\n\n")
            print(self.diffusion_mm_projector.state_dict().keys())

            def get_w(weights, prefix):
                # Adjust to correctly strip the entire prefix including '.' to match model keys
                return {k.replace(prefix + '.', ''): v for k, v in weights.items() if k.startswith(prefix)}

            # Attempt to load weights by stripping the appropriate prefix
            self.diffusion_mm_projector.load_state_dict(get_w(diffusion_mm_projector_weights, 'model.diffusion_mm_projector'))
            print("Loaded pretrained Diffusion adapter!!!")
            # exit()



class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    # def get_dino_vision_tower(self):
    #     return self.get_model().get_vision_tower(load_model = "dino")

    def get_diffusion_vision_tower(self):
        return self.get_model().get_vision_tower(load_model = "diffusion")

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def encode_images_withclip(self, images):
        image_features_clip = self.get_model().get_vision_tower(load_model = "clip")(images)
        image_features_clip = self.get_model().mm_projector(image_features_clip)
        return image_features_clip


    # def encode_images_withdino(self, images):
    #     image_features_dino = self.get_model().get_vision_tower(load_model = "dino")(images)
    #     image_features_dino = self.get_model().dino_mm_projector(image_features_dino)
    #     return image_features_dino

    def encode_images_withdiffusion(self, images):
        image_features_diffusion = self.get_model().get_vision_tower(load_model = "diffusion")(images)
        image_features_diffusion = image_features_diffusion.to(images.device)
        # print(image_features_diffusion.shape)
        image_features_diffusion = self.get_model().diffusion_mm_projector(image_features_diffusion)
        # print(image_features_diffusion.shape)
        return image_features_diffusion

    def prepare_inputs_labels_for_multimodal_withdiffusion(
        self, input_ids, attention_mask, past_key_values, labels, images, diffusion_images
    ):

        

        vision_tower = self.get_vision_tower()


        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
                attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
            return input_ids, attention_mask, past_key_values, None, labels


        # if type(images) is list or images.ndim == 5:
        #     print("I only used multiple images!")
        #     concat_images = torch.cat([image for image in images], dim=0)
        #     image_features = self.encode_images(concat_images)
        #     split_sizes = [image.shape[0] for image in images]
        #     image_features = torch.split(image_features, split_sizes, dim=0)
        #     image_features = [x.flatten(0, 1) for x in image_features]
        # else:
        image_features_clip = self.encode_images_withclip(images)
        # image_features_dino = self.encode_images_withdino(images)
        image_features_diffusion = self.encode_images_withdiffusion(diffusion_images.to(images.device))

        # patch_size = image_features.shape[1]
        # patchify the diffusion features, by repeating them


        new_input_embeds = []
        new_labels = [] if labels is not None else None
        cur_image_idx = 0


        for batch_idx, cur_input_ids in enumerate(input_ids):

            if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:

                half_len = cur_input_ids.shape[0] // 2
                cur_image_features = (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
                cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
                new_input_embeds.append(cur_input_embeds)
                if labels is not None:
                    new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            cur_new_input_embeds = []
            if labels is not None:
                cur_labels = labels[batch_idx]
                cur_new_labels = []
                assert cur_labels.shape == cur_input_ids.shape

            while image_token_indices.numel() > 0:

                cur_image_features_clip = image_features_clip[cur_image_idx]
                # cur_image_features_dino = image_features_dino[cur_image_idx]
                cur_image_features_diffusion = image_features_diffusion[cur_image_idx]

                image_token_start = image_token_indices[0]
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
                        cur_labels = cur_labels[image_token_start+2:]
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))

                    num_patches, clip_dim = cur_image_features_clip.shape

                    clip_dtype = cur_image_features_clip.dtype


                    # Interleave features
                    merged_features = torch.empty(2*num_patches, clip_dim, dtype = clip_dtype)
                    merged_features[0::2] = cur_image_features_clip
                    # merged_features[1::2] = cur_image_features_dino
                    merged_features[1::2] = cur_image_features_diffusion

                    cur_new_input_embeds.append(merged_features)

                    if labels is not None:
                        cur_new_labels.append(cur_labels[:image_token_start])
                        cur_new_labels.append(torch.full((cur_image_features_clip.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        # cur_new_labels.append(torch.full((cur_image_features_dino.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_new_labels.append(torch.full((cur_image_features_diffusion.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
                        cur_labels = cur_labels[image_token_start+1:]
                cur_image_idx += 1
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_input_ids = cur_input_ids[image_token_start+2:]
                else:
                    cur_input_ids = cur_input_ids[image_token_start+1:]

                image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

            if cur_input_ids.numel() > 0:
                if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
                else:
                    cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))

                if labels is not None:
                    cur_new_labels.append(cur_labels)

            cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)


            new_input_embeds.append(cur_new_input_embeds)
            if labels is not None:
                cur_new_labels = torch.cat(cur_new_labels, dim=0)
                new_labels.append(cur_new_labels)

        if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
            max_len = max(x.shape[0] for x in new_input_embeds)

            new_input_embeds_align = []
            for cur_new_embed in new_input_embeds:
                cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
                new_input_embeds_align.append(cur_new_embed)
            new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

            if labels is not None:
                new_labels_align = []
                _new_labels = new_labels
                for cur_new_label in new_labels:
                    cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
                    new_labels_align.append(cur_new_label)
                new_labels = torch.stack(new_labels_align, dim=0)

            if attention_mask is not None:
                new_attention_mask = []
                for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
                    new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
                    new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
                    cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
                    new_attention_mask.append(cur_new_attention_mask)
                attention_mask = torch.stack(new_attention_mask, dim=0)
                assert attention_mask.shape == new_labels.shape
        else:
            new_input_embeds = torch.stack(new_input_embeds, dim=0)
            if labels is not None:
                new_labels  = torch.stack(new_labels, dim=0)

            if attention_mask is not None:
                new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
                assert attention_mask.shape == new_input_embeds.shape[:2]

        return None, attention_mask, past_key_values, new_input_embeds, new_labels


    # def prepare_inputs_labels_for_multimodal_withdino(
    #     self, input_ids, attention_mask, past_key_values, labels, images
    # ):

    #     vision_tower = self.get_vision_tower()


    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
    #             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
    #         return input_ids, attention_mask, past_key_values, None, labels


    #     if type(images) is list or images.ndim == 5:
    #         print("I only used multiple images!")
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         image_features = [x.flatten(0, 1) for x in image_features]
    #     else:
    #         image_features_clip = self.encode_images_withclip(images)
    #         image_features_dino = self.encode_images_withdino(images)


    #     new_input_embeds = []
    #     new_labels = [] if labels is not None else None
    #     cur_image_idx = 0


    #     for batch_idx, cur_input_ids in enumerate(input_ids):

    #         if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:

    #             half_len = cur_input_ids.shape[0] // 2
    #             cur_image_features = (0. * self.get_model().mm_projector(vision_tower.dummy_feature)).sum()
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
    #             cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             if labels is not None:
    #                 new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue

    #         image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

    #         cur_new_input_embeds = []
    #         if labels is not None:
    #             cur_labels = labels[batch_idx]
    #             cur_new_labels = []
    #             assert cur_labels.shape == cur_input_ids.shape

    #         while image_token_indices.numel() > 0:

    #             cur_image_features_clip = image_features_clip[cur_image_idx]
    #             cur_image_features_dino = image_features_dino[cur_image_idx]

    #             image_token_start = image_token_indices[0]
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
    #                 if labels is not None:
    #                     cur_new_labels.append(cur_labels[:image_token_start])
    #                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
    #                     cur_labels = cur_labels[image_token_start+2:]
    #             else:
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))

    #                 num_patches, clip_dim = cur_image_features_clip.shape

    #                 clip_dtype = cur_image_features_clip.dtype


    #                 # Interleave features
    #                 merged_features = torch.empty(2*num_patches, clip_dim, dtype = clip_dtype)
    #                 merged_features[0::2] = cur_image_features_clip
    #                 merged_features[1::2] = cur_image_features_dino

    #                 cur_new_input_embeds.append(merged_features)

    #                 if labels is not None:
    #                     cur_new_labels.append(cur_labels[:image_token_start])
    #                     cur_new_labels.append(torch.full((cur_image_features_clip.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_new_labels.append(torch.full((cur_image_features_dino.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_labels = cur_labels[image_token_start+1:]
    #             cur_image_idx += 1
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_input_ids = cur_input_ids[image_token_start+2:]
    #             else:
    #                 cur_input_ids = cur_input_ids[image_token_start+1:]

    #             image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]

    #         if cur_input_ids.numel() > 0:
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
    #             else:
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))

    #             if labels is not None:
    #                 cur_new_labels.append(cur_labels)

    #         cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)


    #         new_input_embeds.append(cur_new_input_embeds)
    #         if labels is not None:
    #             cur_new_labels = torch.cat(cur_new_labels, dim=0)
    #             new_labels.append(cur_new_labels)

    #     if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
    #         max_len = max(x.shape[0] for x in new_input_embeds)

    #         new_input_embeds_align = []
    #         for cur_new_embed in new_input_embeds:
    #             cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
    #             new_input_embeds_align.append(cur_new_embed)
    #         new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

    #         if labels is not None:
    #             new_labels_align = []
    #             _new_labels = new_labels
    #             for cur_new_label in new_labels:
    #                 cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
    #                 new_labels_align.append(cur_new_label)
    #             new_labels = torch.stack(new_labels_align, dim=0)

    #         if attention_mask is not None:
    #             new_attention_mask = []
    #             for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
    #                 new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
    #                 new_attention_mask.append(cur_new_attention_mask)
    #             attention_mask = torch.stack(new_attention_mask, dim=0)
    #             assert attention_mask.shape == new_labels.shape
    #     else:
    #         new_input_embeds = torch.stack(new_input_embeds, dim=0)
    #         if labels is not None:
    #             new_labels  = torch.stack(new_labels, dim=0)

    #         if attention_mask is not None:
    #             new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #             attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
    #             assert attention_mask.shape == new_input_embeds.shape[:2]

    #     return None, attention_mask, past_key_values, new_input_embeds, new_labels

    # def prepare_inputs_labels_for_multimodal(
    #     self, input_ids, attention_mask, past_key_values, labels, images
    # ):

    #     print("I am here")
    #     exit()
    #     vision_tower = self.get_vision_tower()
    #     if vision_tower is None or images is None or input_ids.shape[1] == 1:
    #         if past_key_values is not None and vision_tower is not None and images is not None and input_ids.shape[1] == 1:
    #             attention_mask = torch.ones((attention_mask.shape[0], past_key_values[-1][-1].shape[-2] + 1), dtype=attention_mask.dtype, device=attention_mask.device)
    #         return input_ids, attention_mask, past_key_values, None, labels

    #     if type(images) is list or images.ndim == 5:
    #         concat_images = torch.cat([image for image in images], dim=0)
    #         image_features = self.encode_images(concat_images)
    #         split_sizes = [image.shape[0] for image in images]
    #         image_features = torch.split(image_features, split_sizes, dim=0)
    #         image_features = [x.flatten(0, 1) for x in image_features]
    #     else:
    #         image_features = self.encode_images(images)

    #     new_input_embeds = []
    #     new_labels = [] if labels is not None else None
    #     cur_image_idx = 0
    #     for batch_idx, cur_input_ids in enumerate(input_ids):
    #         if (cur_input_ids == IMAGE_TOKEN_INDEX).sum() == 0:
    #             # multimodal LLM, but the current sample is not multimodal
    #             # FIXME: this is a hacky fix, for deepspeed zero3 to work
    #             half_len = cur_input_ids.shape[0] // 2
    #             cur_image_features = image_features[cur_image_idx]
    #             cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids[:half_len])
    #             cur_input_embeds_2 = self.get_model().embed_tokens(cur_input_ids[half_len:])
    #             cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_input_embeds_2], dim=0)
    #             new_input_embeds.append(cur_input_embeds)
    #             if labels is not None:
    #                 new_labels.append(labels[batch_idx])
    #             cur_image_idx += 1
    #             continue
    #         image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    #         cur_new_input_embeds = []
    #         if labels is not None:
    #             cur_labels = labels[batch_idx]
    #             cur_new_labels = []
    #             assert cur_labels.shape == cur_input_ids.shape
    #         while image_token_indices.numel() > 0:
    #             cur_image_features = image_features[cur_image_idx]
    #             image_token_start = image_token_indices[0]
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start-1]).detach())
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start-1:image_token_start]))
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[image_token_start+1:image_token_start+2]))
    #                 if labels is not None:
    #                     cur_new_labels.append(cur_labels[:image_token_start])
    #                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_new_labels.append(cur_labels[image_token_start:image_token_start+1])
    #                     cur_labels = cur_labels[image_token_start+2:]
    #             else:
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids[:image_token_start]))
    #                 cur_new_input_embeds.append(cur_image_features)
    #                 if labels is not None:
    #                     cur_new_labels.append(cur_labels[:image_token_start])
    #                     cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=labels.device, dtype=labels.dtype))
    #                     cur_labels = cur_labels[image_token_start+1:]
    #             cur_image_idx += 1
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_input_ids = cur_input_ids[image_token_start+2:]
    #             else:
    #                 cur_input_ids = cur_input_ids[image_token_start+1:]
    #             image_token_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0]
    #         if cur_input_ids.numel() > 0:
    #             if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids).detach())
    #             else:
    #                 cur_new_input_embeds.append(self.get_model().embed_tokens(cur_input_ids))
    #             if labels is not None:
    #                 cur_new_labels.append(cur_labels)
    #         cur_new_input_embeds = [x.to(device=self.device) for x in cur_new_input_embeds]
    #         cur_new_input_embeds = torch.cat(cur_new_input_embeds, dim=0)
    #         new_input_embeds.append(cur_new_input_embeds)
    #         if labels is not None:
    #             cur_new_labels = torch.cat(cur_new_labels, dim=0)
    #             new_labels.append(cur_new_labels)

    #     if any(x.shape != new_input_embeds[0].shape for x in new_input_embeds):
    #         max_len = max(x.shape[0] for x in new_input_embeds)

    #         new_input_embeds_align = []
    #         for cur_new_embed in new_input_embeds:
    #             cur_new_embed = torch.cat((cur_new_embed, torch.zeros((max_len - cur_new_embed.shape[0], cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)), dim=0)
    #             new_input_embeds_align.append(cur_new_embed)
    #         new_input_embeds = torch.stack(new_input_embeds_align, dim=0)

    #         if labels is not None:
    #             new_labels_align = []
    #             _new_labels = new_labels
    #             for cur_new_label in new_labels:
    #                 cur_new_label = torch.cat((cur_new_label, torch.full((max_len - cur_new_label.shape[0],), IGNORE_INDEX, dtype=cur_new_label.dtype, device=cur_new_label.device)), dim=0)
    #                 new_labels_align.append(cur_new_label)
    #             new_labels = torch.stack(new_labels_align, dim=0)

    #         if attention_mask is not None:
    #             new_attention_mask = []
    #             for cur_attention_mask, cur_new_labels, cur_new_labels_align in zip(attention_mask, _new_labels, new_labels):
    #                 new_attn_mask_pad_left = torch.full((cur_new_labels.shape[0] - labels.shape[1],), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 new_attn_mask_pad_right = torch.full((cur_new_labels_align.shape[0] - cur_new_labels.shape[0],), False, dtype=attention_mask.dtype, device=attention_mask.device)
    #                 cur_new_attention_mask = torch.cat((new_attn_mask_pad_left, cur_attention_mask, new_attn_mask_pad_right), dim=0)
    #                 new_attention_mask.append(cur_new_attention_mask)
    #             attention_mask = torch.stack(new_attention_mask, dim=0)
    #             assert attention_mask.shape == new_labels.shape
    #     else:
    #         new_input_embeds = torch.stack(new_input_embeds, dim=0)
    #         if labels is not None:
    #             new_labels  = torch.stack(new_labels, dim=0)

    #         if attention_mask is not None:
    #             new_attn_mask_pad_left = torch.full((attention_mask.shape[0], new_input_embeds.shape[1] - input_ids.shape[1]), True, dtype=attention_mask.dtype, device=attention_mask.device)
    #             attention_mask = torch.cat((new_attn_mask_pad_left, attention_mask), dim=1)
    #             assert attention_mask.shape == new_input_embeds.shape[:2]

    #     return None, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False