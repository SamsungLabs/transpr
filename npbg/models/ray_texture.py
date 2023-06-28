import torch
import torch.nn as nn
from npbg.models.texture import Texture


class PointTexture(Texture):
    def __init__(self, num_channels, size, checkpoint=None, init_method='zeros', alpha_dim=1, alpha_init_scale=0.1, reg_weight=0., requires_grad=True, density_scale=None):
        super().__init__()

        assert isinstance(size, int), 'size must be int'

        shape = 1, num_channels, size

        if checkpoint:
            self.texture_ = torch.load(checkpoint, map_location='cpu')['texture'].texture_
        else:
#             print(init_method)
            if init_method == 'rand':
                texture = torch.rand(shape)
            elif init_method == 'zeros':
                texture = torch.zeros(shape)
            elif init_method == 'alpha_const':                
                texture = torch.cat([torch.rand(1, num_channels-alpha_dim, size), alpha_init_scale*torch.ones(1, alpha_dim, size)], axis=1)
            else:
                raise ValueError(init_method)
            self.texture_ = nn.Parameter(texture.float(), requires_grad=requires_grad)
        
        self.reg_weight = reg_weight
        if density_scale is not None:
            self.density_scale = nn.Parameter(torch.FloatTensor([density_scale]), requires_grad=requires_grad)

    def null_grad(self):
        self.texture_.grad = None

    def reg_loss(self):
        return self.reg_weight * torch.mean(torch.pow(self.texture_, 2))

    def forward(self, inputs):
        
        ids = inputs 
        sh = ids.shape # max_ray_length x N rays
        n_pts = self.texture_.shape[-1]
        
        ind = ids.contiguous().view(-1).long() # max_ray_length * N rays

        texture = self.texture_.permute(1, 0, 2) # Cx1xN points
#         texture = texture.expand(texture.shape[0], sh[0], texture.shape[2]) # C x B x N points
#         print('texture', texture.shape)
        texture = texture.contiguous().view(texture.shape[0], -1) # C x B*N points
    
        sample = torch.index_select(texture, 1, ind) # C x max_ray_length * N rays
        sample = sample.contiguous().view(sample.shape[0], *sh) # C x max_ray_length x N rays
        
        
        # C x max_ray_length x N rays - >  max_ray_length x N rays x C
        sample = sample.permute(1, 2, 0) 

        return sample


