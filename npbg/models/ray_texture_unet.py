import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from npbg.models.dynamic_module import DynamicModule
from npbg.utils.train import save_model, load_model_checkpoint
from npbg.datasets.common import RGBA


class RayTexture(nn.Module):
    def __init__(self, net, ray_block, renderer, textures, scenes, crop_size, output_dim, num_output_channels,   ordered_keys=[], **kwargs):
        super().__init__()
        
        self.net = net
        self.scenes = DynamicModule(scenes, ordered_keys)
        self.textures = DynamicModule(textures, ordered_keys)
        self.ray_block = ray_block
        self.renderer = renderer
        self.width = crop_size[0]
        self.height = crop_size[1]
        self.scales = renderer.sampler.scales
        self.output_dim = output_dim # dimensionality of rasterized descriptors: pseudo-colors (+ alpha)
        self.max_ray_length = self.renderer.sampler.max_ray_length
        self.num_output_channels = num_output_channels # rendering network output: RGB(A)
    
    def reg_loss(self):
        loss = 0
        for tid in self.textures._modules:
            loss += self.textures._modules[tid].reg_loss()

        return loss    

      
    def load(self, save_dir, epoch, stage=0):
        for name, texture in self.textures.submodules.items():
            self._load_model(texture, save_dir, epoch, stage, name)
        self._load_model(self.net, save_dir, epoch, stage, 'net')
        
    def _load_model(self, model, save_dir, epoch, stage, name):  
        model_class = model.__class__.__name__
        
        filename = f'{model_class}_stage_{stage}_epoch_{epoch}_{name}.pth'     
         
        save_path = os.path.join(save_dir, filename)
        print('loading', save_path)
        if os.path.exists(save_path):     
            model, args = load_model_checkpoint(save_path, model)
        else:
            print(f"No checkpoint for {model_class} in {save_dir}")

        
    def dataset_load(self, ds_ids):
        self.textures.load_modules(ds_ids)
        self.scenes.load_modules(ds_ids)
        
    def dataset_unload(self):
        self.textures.unload_modules()
        self.scenes.unload_modules()
                
    def _render_tensors(self, batches, texture, canvas_indices, rays_lengths, alpha_jitter=None, density_scale=None, transform_alpha=True):
        
        rendered_textures = {}
        holes_masks = {}
        
        for i, scale in enumerate(self.scales):  
                
            canvas_index = canvas_indices[i]                 
            batch = batches[i] # max_ray_length x H x W
            rays_lengths_batch = rays_lengths[i]    
           
            canvas_mask = torch.zeros((1, self.height // scale, self.width // scale), device=batch.device)
        
           
            if batch.numel()==0 or rays_lengths_batch.numel()==0 or canvas_index.numel()==0:
                
                rendered_textures[scale] = torch.zeros((1, self.output_dim, self.height // scale, self.width // scale), device=batch.device)
                
                holes_masks[scale] = canvas_mask[None] 
                continue

            texture_sample = texture(batch)
            
            canvas_mask.index_put_((torch.zeros_like(canvas_index[:, 0]), canvas_index[:, 0], canvas_index[:, 1]), torch.ones(canvas_index.shape[0], dtype=canvas_mask.dtype, device=canvas_mask.device))
            
            canvas_texture = torch.zeros((self.height // scale, self.width // scale, self.output_dim), device=batch.device)
            
            if self.ray_block is not None:  
                render = self.ray_block(texture_sample, rays_lengths_batch, alpha_jitter=alpha_jitter, density_scale=density_scale, transform_alpha=transform_alpha)                
            else:  
                render = texture_sample[0]
    
            canvas_texture.index_put_((canvas_index[:,0], canvas_index[:,1]), render)
            canvas_texture = canvas_texture.permute(2, 0, 1)                
                
                
            # treating like batch 1            
            rendered_textures[scale] = canvas_texture[None]
            holes_masks[scale] = canvas_mask[None]
        
        return rendered_textures, holes_masks
        
    def forward(self, batch, transform_alpha=True, target=None, clamp_output=False, **kwargs):
        '''
        batch: view_matrix, proj_matrix, id
        '''
        batch_size = len(batch['view_matrix'])
        
        ids = batch['id']

        if torch.is_tensor(ids):
            ids = [id_.item() for id_ in ids.cpu()]
        
        out = []

        out_inputs_tmp = defaultdict(list)
        out_masks_tmp = defaultdict(list)
        
        for i in range(batch_size):
            view_matrix = batch['view_matrix'][i]
            proj_matrix = batch['proj_matrix'][i]            
            id_ = ids[i]
                        
            scene = self.scenes._modules[self.scenes.id_module_map[id_]]
            texture_model = self.textures._modules[self.textures.id_module_map[id_]] 
            density_scale = texture_model.density_scale.item() if hasattr(texture_model, 'density_scale') else None
            alpha_jitter = batch['alpha_jitter'][i] if 'alpha_jitter' in batch else None

            ids_to_render = scene.ids.to(texture_model.texture_.device)
            
            # projection 
            inputs = self.renderer.render(scene, ids_to_render, view_matrix, proj_matrix) # batches, rays_lengths, canvas_indices
            
            inputs, masks = self._render_tensors(**inputs, texture=texture_model, alpha_jitter=alpha_jitter, density_scale=density_scale, transform_alpha=transform_alpha)
            
            for idx, i_scale in enumerate(inputs):
                out_inputs_tmp[i_scale].append(inputs[i_scale])
                out_masks_tmp[i_scale].append(masks[i_scale])
        
        for i_scale in out_inputs_tmp:
            out_inputs_tmp[i_scale] = torch.cat(out_inputs_tmp[i_scale], 0)
            out_masks_tmp[i_scale] = torch.cat(out_masks_tmp[i_scale], 0)

        target_out = None
        if target is not None: 
            target_out = torch.zeros_like(target)
        out_inputs = [torch.zeros_like(out_input) for i_scale, out_input in out_inputs_tmp.items()]
        out_masks = list(out_masks_tmp.values())

        overlay_index = batch['overlay_index'] if 'overlay_index' in batch else None

        for i in range(batch_size):
            overlay_index_ = overlay_index[i].item() 
            if overlay_index_ <0: 
                if target is not None:
                    target_out[i] = target[i]
                for j, scale in enumerate(self.scales):
                    out_inputs[j][i] = out_inputs_tmp[scale][i]
            else:
                if target is not None:
                    blend_mask = (1 - target[i][-1:])*target[overlay_index_][-1:]
                    target_out[i][:-1] = target[i][:-1] + target[overlay_index_][:-1]*blend_mask
                    target_out[i][-1:] = 1-(1-target[overlay_index_][-1:])*(1-target[i][-1:])

                for j, scale in enumerate(self.scales):
                    blend_mask_input = (1-out_inputs_tmp[scale][i][-1:])*out_inputs_tmp[scale][overlay_index_][-1:]
                    out_inputs[j][i][:-1] = out_inputs_tmp[scale][i][:-1] + out_inputs_tmp[scale][overlay_index_][:-1]*blend_mask_input
                    out_inputs[j][i][-1:] = 1-(1-out_inputs_tmp[scale][overlay_index_][-1:])*(1-out_inputs_tmp[scale][i][-1:])
        

        net_output = self.net(*out_inputs)
                        
        if clamp_output:
            net_output = torch.clamp_(net_output, 0, 1)

        out_data = {}

        if self.num_output_channels==RGBA and ('background' in batch):
            assert net_output.shape[1]==RGBA        
            net_output[:,:-1] = net_output[:,:-1] + batch['background']*(1-net_output[:,-1:])

            if target_out is not None:
                target_out[:,:-1] = target_out[:,:-1] + batch['background']*(1-target_out[:,-1:])
                out_data['target'] = target_out  

        assert out_masks[0].shape[0]==out_inputs[0].shape[0]==net_output.shape[0]

        out_data['net_output'] = net_output                   
        
        if 'return_inputs' in kwargs and kwargs['return_inputs']:
            out_data['inputs'] = out_inputs
        if 'return_masks' in kwargs and kwargs['return_masks']:
            out_data['masks'] = out_masks            
        
        return out_data