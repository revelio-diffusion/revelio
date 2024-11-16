import torch
import torch.nn as nn
from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
from typing import Callable
from contextlib import contextmanager
from typing import List, Union, Dict, Tuple
from functools import partial
from torch.nn import functional as F
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
import sys
import os


class Hook():
  def __init__(self, module_name: str, hook_fn: Callable, return_module_output = True):

    self.return_module_output = return_module_output
    self.function = self.get_full_hook_fn(hook_fn)
    self.attr_path = module_name

  def get_full_hook_fn(self, hook_fn: Callable):

    def full_hook_fn(module, module_input, module_output):
      hook_fn_output = hook_fn(module_output)
      if self.return_module_output:
        return module_output
      else:
        return hook_fn_output 

    return full_hook_fn
  
  
  def get_module(self, model):
    return self.get_nested_attr(model, self.attr_path)

  def get_nested_attr(self, model, attr_path):

    module = model
    attributes = attr_path.split(".")
    for attr in attributes:
        if '[' in attr:
            attr_name, index = attr[:-1].split('[')
            module = getattr(module, attr_name)[int(index)]
        else:
            module = getattr(module, attr)
    return module



class HookedStableDiffusion():
  def __init__(self, model_name: str, image_width:int, device = 'cuda'):
    model, tokenizer, vae, text_encoder, noise_scheduler = self.get_model(model_name, image_width)
    self.model = model.to(device)
    self.tokenizer = tokenizer
    self.vae = vae.to(device)
    self.text_encoder = text_encoder.to(device)
    self.noise_scheduler = noise_scheduler

  def get_model(self, model_name, image_width):
    model = UNet2DConditionModel.from_pretrained(model_name, subfolder="unet").requires_grad_(False)
    tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder="tokenizer", revision=None)
    vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae", revision=None, variant=None).requires_grad_(False)
    text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    noise_scheduler = DDPMScheduler.from_pretrained(model_name, subfolder="scheduler")
    return model, tokenizer, vae, text_encoder, noise_scheduler

  def run_with_cache(self, list_of_hook_locations: List[Tuple[int,str]], *args, return_type = "output", **kwargs):
    cache_dict, list_of_hooks = self.get_caching_hooks(list_of_hook_locations)
    with self.hooks(list_of_hooks) as hooked_model:
      with torch.no_grad():
        output = hooked_model(**kwargs)
        
    if return_type=="output":
      return output, cache_dict

    else:
      raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")

  def get_caching_hooks(self, list_of_hook_locations: List[Tuple[int,str]]):

    cache_dict = {}
    list_of_hooks=[]
    def save_activations(name, activations):
      cache_dict[name] = activations.detach()
    for ( module_name) in list_of_hook_locations:
      hook_fn = partial(save_activations, (module_name))
      hook = Hook( module_name, hook_fn)
      list_of_hooks.append(hook)
    return cache_dict, list_of_hooks

  @torch.no_grad
  def run_with_hooks(self, list_of_hooks: List[Hook], *args, return_type = "output", **kwargs):
    with self.hooks(list_of_hooks) as hooked_model:
      with torch.no_grad():

        output = hooked_model(*args)

    if return_type=="output":
      return output
    else:
      raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output'.")
    

  @contextmanager
  def hooks(self, hooks: List[Hook]):

    hook_handles = []
    try:
      for hook in hooks:

        module = hook.get_module(self.model)
        handle = module.register_forward_hook(hook.function)
        hook_handles.append(handle)
      yield self.model
    finally:
      for handle in hook_handles:
        handle.remove()
            
  def to(self, device):
    self.model = self.model.to(device)

  def __call__(self, *args, return_type = 'output', **kwargs):
    return self.forward(*args, return_type = return_type, **kwargs)

  def forward(self, *args, return_type = 'output', **kwargs):
    if return_type=='output':
      return self.model(*args, **kwargs)
    elif return_type == 'loss':
      output = self.model(*args, **kwargs)
      return self.contrastive_loss(output.logits_per_image, output.logits_per_text)
    else:
      raise Exception(f"Unrecognised keyword argument return_type='{return_type}'. Must be either 'output' or 'loss'.")
  
  def eval(self):
    self.model.eval()
    
  def train(self):
    self.model.train()
