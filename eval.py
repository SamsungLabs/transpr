import numpy as np
import cv2

import os, sys
from os.path import join
from tqdm import tqdm
from munch import munchify

import torch

from npbg.utils.arguments import MyArgumentParser, eval_args
from npbg.utils.train import get_module, freeze, load_model_checkpoint
from npbg.data.io.yaml import load_yaml

from npbg.gl.utils import get_proj_matrix

from npbg.utils.arguments import eval_modules
from npbg.datasets.common import RGBA, RGB, write_image

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def evaluate(pipeline, args):

    pipeline.model.cuda().eval() # CPU rendering not supported due to sorting stability issues
    
    save_dir = os.path.join(args.save_dir, args.dataset_names[0])
    os.makedirs(save_dir, exist_ok=True)   
        
    dataset = pipeline.ds_test[0]    
    loader = DataLoader(dataset, batch_size=1, drop_last=False, pin_memory=False, shuffle=False)

    transform_alpha = not args.raw_alpha
    num_save_channels = RGBA if args.save_with_alpha and args.num_output_channels==RGBA else RGB   
    
    with torch.no_grad():
        renders = []
        for idx, batch in enumerate(tqdm(loader)):

            batch['input']['alpha_jitter'] = torch.FloatTensor([args.alpha_jitter])
            
            for k in batch['input']:
                if isinstance(batch['input'][k], torch.Tensor):
                    batch['input'][k]=batch['input'][k].cuda()

            output = pipeline.model(batch['input'], transform_alpha=transform_alpha)

            net_output = output['net_output']

            img = net_output[0].detach().cpu().permute(1,2,0).numpy()
            write_image(img[:,:,:num_save_channels], join(save_dir, f'{idx}.png'))
            
            del output               
            torch.cuda.empty_cache()    
    

if __name__ == '__main__':
    parser = MyArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument
    parser.add('--config', type=str)
    parser.add('--save_dir', type=str)
    parser.add('--save_with_alpha', action='store_true')
    parser.add('--raw_alpha', action='store_true')   
    parser.add('--alpha_jitter', type=float, default=1.0)

    
    infer_args, _ = parser.parse_known_args()
    config = load_yaml(infer_args.config)

    net_checkpoint = torch.load(config['net_ckpt']) 

    args_ = net_checkpoint['args']
    args_['inference'] = True    
    args_.update(vars(infer_args))
    args_.update(config)
    eval_modules(args_)        
    args = munchify(args_)
    
    
    pipeline = get_module(args.pipeline)()
    pipeline.create(args)

    pipeline.dataset_load(pipeline.ds_test)

    load_model_checkpoint(config['texture_ckpt'], pipeline.model.textures.submodules[config['dataset_name']])
    load_model_checkpoint(config['net_ckpt'], pipeline.model.net)    

    evaluate(pipeline, args)
    
    pipeline.dataset_unload(pipeline.ds_test)
    

