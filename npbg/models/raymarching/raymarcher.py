import torch
import torch.nn as nn


    
class Raymarcher(nn.Module):
    def __init__(self, in_features, use_alpha=True, alpha_sigmoid=False, alpha_clamp=False, **kwargs):
        super().__init__()

        self.output_dim = in_features
        self.relu = nn.ReLU()
        self.use_alpha = use_alpha

        self.alpha_sigmoid = alpha_sigmoid
        self.alpha_clamp = alpha_clamp

    def forward(self, batch, ray_lengths, alpha_jitter=None, density_scale=None, transform_alpha=True, **kwargs):
        
        L, N, C = batch.shape     
        F = C-1
        
        features = batch[:,:,:F]
        alpha =  batch[:,:,F:]
          
        if transform_alpha:                
            if self.alpha_clamp:
                alpha = torch.clamp(alpha, 0, 1)
            elif self.alpha_sigmoid:
                alpha = torch.sigmoid(alpha)
            else:
                alpha = self.relu(alpha)
                alpha = nn.functional.tanh(alpha)
        
        if alpha_jitter is not None:
            if density_scale is not None:
                alpha_jitter = alpha_jitter**density_scale
            alpha = alpha*alpha_jitter
    
        canvas = torch.zeros_like(features[0]) # N x C

        
        alpha_dst = torch.ones_like(alpha[0])

        for step in range(L):
            alpha_src = alpha[step] 

            alpha_mask = (step<ray_lengths).float()[...,None]

            canvas = alpha_dst*(alpha_mask*alpha_src*features[step]) +  canvas
            alpha_dst = (1-alpha_src*alpha_mask)*alpha_dst


        if self.use_alpha:
            alpha_out = 1-alpha_dst
            canvas = torch.cat([canvas, alpha_out], dim=1)
        
        return canvas    
