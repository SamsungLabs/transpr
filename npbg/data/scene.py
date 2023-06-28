from torch import nn
import torch

class XYZScene(nn.Module):
    def __init__(self, scene_data, **kwargs):
        super().__init__()
        
        self.xyz = torch.FloatTensor(scene_data['pointcloud']['xyz'])
        self.rgb = torch.FloatTensor(scene_data['pointcloud']['rgb']) if 'rgb' in scene_data['pointcloud'] else None
        self.ids = torch.arange(self.vertices.shape[0]).float()
        
        self.id = kwargs['id'] if 'id' in kwargs else None 
    
    @property
    def vertices(self):        
        return torch.cat([self.xyz, torch.ones_like(self.xyz[:,[0]]).to(self.xyz.device)], dim=1)