import torch.nn.functional as F
import torch.nn as nn
import torch

from npbg.criterions.vgg_loss import VGGLoss


class VGGL1Alpha(nn.Module):
    def __init__(self, use_mask=False, l1_weight=400.0):
        super().__init__()
        self.l1 = nn.L1Loss()   
        self.vgg = VGGLoss(partialconv=False)
        self.use_mask = use_mask
        self.l1_weight = l1_weight
        
    def forward(self, output_, target):
        if isinstance(output_, dict):
            output = output_['net_output']
            if 'target' in output_:
                target = output_['target']
        else:
            output = output_

        assert output.shape[1]==4
        assert target.shape[1]==4
        
        vgg_loss =  self.vgg(output[:,:3], target[:,:3])
        
        if self.l1_weight>0:
            l1_loss = self.l1(output[:,[3]], target[:,[3]])
            vgg_loss = vgg_loss + self.l1_weight*l1_loss
        
        return vgg_loss
            
        
class L1(nn.Module):
    def __init__(self, use_mask=False, l1_weight=0.0):
        super().__init__()
        self.l1 = nn.L1Loss()   
        self.use_mask = use_mask
        self.l1_weight = l1_weight
        
    def forward(self, output, target):
        assert output.shape[1]==4
        assert target.shape[1]==4
        
        l1_rgb =  self.l1(output[:,:3], target[:,:3])
        
        if self.l1_weight>0:
            l1_alpha = self.l1(output[:,[3]], target[:,[3]])
            l1_rgb = l1_rgb + self.l1_weight*l1_alpha
        
        return l1_rgb 