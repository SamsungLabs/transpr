import torch
from torch import nn


   
class Renderer(nn.Module):
    def __init__(self, projector, sampler):
        super().__init__()

        self.projector = projector
        self.sampler = sampler
        
    def render(self, scene, pts_meta, view_matrix, proj_matrix, debug=False):
        screen_coords, depth, exclude_mask = self.projector(scene.vertices.to(pts_meta.device), view_matrix, proj_matrix)
        depth_t_masked = depth[exclude_mask]

        if depth_t_masked.numel()==0:                
            return {'batches': [torch.empty(0).to(pts_meta.device) for scale in self.sampler.scales],  \
                    'rays_lengths': [torch.empty(0).to(pts_meta.device) for scale in self.sampler.scales],\
                    'canvas_indices': [torch.empty(0).to(pts_meta.device) for scale in self.sampler.scales]}
        
        coords_t_masked =  screen_coords[exclude_mask] # 1 x n_vertices x 1 x 2, keep dims 
        
        pts_meta_t_masked = torch.masked_select(pts_meta, exclude_mask)
        
        batches, rays_lengths, canvas_indices  = self.sampler(coords_t_masked, depth_t_masked, pts_meta_t_masked)
        
        return {'batches': batches, 'rays_lengths': rays_lengths, 'canvas_indices': canvas_indices}

    
    