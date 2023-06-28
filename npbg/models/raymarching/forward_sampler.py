import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# The following decorator definition can be optionally commented to enable line_profiler support for the module:
def profile(f):
    return f
# from pytorch_memlab import profile



class ForwardSamplerMultiscale3d(nn.Module):

    def __init__(self, OH, OW, scales=[1], max_ray_length=100):
        super().__init__()
        # output image height and width
        self.OH = OH 
        self.OW = OW

        # a list of monotonically increasing powers of 2 (e. g. 5 scales: [1, 2, 4, 8, 16])
        self.scales = scales    
        
        self.max_ray_length = max_ray_length
        

    @profile
    def forward(self, coords, depth, pts_meta=None):
        '''
        coords: screen-space coordinates
        depth: raw depth (z)
        pts_meta: indices of corresponding points
        '''
        # Triple sorting performed corresponds to the sorting of tensor by (I, J, D) tuple key for each sample.
        # This way, pts_meta is now split into the (I, J) consecutive sections sorted by D values.
        # ---
        n_points = coords.shape[0]
        
        canvas_indices = [] # indices to map projected vertices onto a canvas of a fixed resolution
        batches = [] # 3d tensors for further alpha blending
        rays_lengths = [] # lengths of rays in batches

        # preparation of point cloud for the first scale
        height_first = self.OH // self.scales[0]
        width_first = self.OW // self.scales[0]

        coords[:, 0] *= height_first
        coords[:, 1] *= width_first 
        
        coords[:, 0].clamp_(0, height_first - 1)
        coords[:, 1].clamp_(0, width_first - 1)
        coords = torch.floor(coords)  

        coords = coords.long() 

        # sorting by raw depth (D)
        argsort = torch.argsort(depth, descending=False) 

        pts_meta = pts_meta[argsort]
        coords = coords[argsort]
        depth = depth[argsort]

        pts_meta_orig = pts_meta
        coords_orig = coords

        for i, scale in enumerate(self.scales):
            width = width_first//scale
            height = height_first//scale
            pts_meta = pts_meta_orig.clone()
            coords = coords_orig.clone() // scale
            
            coords[:, 0].clamp_(0, height - 1)
            coords[:, 1].clamp_(0, width - 1)

            # sorting by Wi+J
            argsort = torch.argsort(width * coords[:, -2] + coords[:, -1])
            pts_meta = pts_meta[argsort]
            coords = coords[argsort]

            i_col = coords[:, -2]
            j_col = coords[:, -1]

            # grouping points into rays according to the correspondence to a particular pixel (i, j)          
            if n_points == 1:
                change_mask = torch.ones_like(i_col).bool()                
            else:
                change_mask = (i_col[1:] != i_col[:-1]) | (j_col[1:] != j_col[:-1])                
                change_mask = torch.cat([torch.ones(1, device=coords.device, dtype=torch.bool), change_mask])

            canvas_index = torch.stack([i_col[change_mask], j_col[change_mask]], dim=1)
            canvas_indices.append(canvas_index.long())

            change_mask_cumsum = change_mask.long().cumsum(0)
            section_start_inds = torch.nonzero(change_mask)[:, 0]  # tensor with column indices corresponding to changed (i, j) pairs

            section_start_inds = torch.cat([torch.zeros(1, device=coords.device, dtype=torch.long), section_start_inds], dim=0)         
            
            cumsum_index = section_start_inds[change_mask_cumsum]

            index_in_group = torch.arange(change_mask.shape[0], device=coords.device) - cumsum_index

            nonzero_change = section_start_inds[1:]   

            if nonzero_change.numel() == 0: 
                nonzero_change = torch.tensor([nonzero_change], device=coords.device)

            rays_end = torch.tensor([change_mask.shape[0]], device=coords.device, dtype=nonzero_change.dtype)
            
            nonzero_change = torch.cat([nonzero_change, rays_end])

            if nonzero_change.numel() == 1:
                rays_lengths_batch = nonzero_change.squeeze()
            else:
                rays_lengths_batch = nonzero_change[1:] - nonzero_change[:-1]

            # truncating rays longer than max_ray_length
            rays_lengths_batch[rays_lengths_batch>self.max_ray_length] = self.max_ray_length
            
            # max_ray_length x # of non-emtpy rays
            batch = torch.zeros((self.max_ray_length, rays_lengths_batch.shape[0]), device=coords.device)
            
            assert canvas_index.shape[0]==rays_lengths_batch.shape[0]

            change_mask_cumsum = torch.clamp_min(change_mask_cumsum-1, 0)
            
            index_mask = index_in_group<self.max_ray_length
            pts_meta = pts_meta[index_mask]
            index_in_group = index_in_group[index_mask]
            change_mask = change_mask[index_mask]            


            change_mask_cumsum = change_mask_cumsum[index_mask]
            
            batch.index_put_((index_in_group, change_mask_cumsum), pts_meta)

            batches.append(batch)
            rays_lengths.append(rays_lengths_batch)
        
        
        return batches, rays_lengths, canvas_indices
