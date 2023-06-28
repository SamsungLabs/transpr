import os
from pathlib import Path
from torch import optim
import numpy as np

from torch.utils.data import DataLoader

from npbg.pipelines import Pipeline
from npbg.models.unet import UNet

from npbg.datasets.image import ImageDatasetFactory
from npbg.data.scene import XYZScene
from npbg.data.io.yaml import load_yaml
from npbg.data.io.ply import read_pc
from npbg.datasets.common import get_dataset_config
from npbg.utils.train import freeze

from npbg.gl.utils import intrinsics_from_xml, fix_relative_path
from npbg.utils.train import load_model_checkpoint

TextureOptimizerClass = optim.RMSprop



class RayPipeline(Pipeline):
    def __init__(self):
        pass

    def export_args(self, parser):
        # add arguments of this pipeline to the cmd parser
        parser.add_argument('--descriptor_size', type=int, default=8)
        parser.add_argument('--texture_size', type=int)
        parser.add_argument('--texture_ckpt', type=Path)
        parser.add('--texture_lr', type=float, default=1e-1)
        parser.add('--num_mipmap', type=int, default=5)
        parser.add('--max_ray_length', type=int, default=100)
        parser.add('--inference', action='store_bool', default=False)  
        parser.add('--num_output_channels', type=int, default=3)
        parser.add('--conv_block', type=str, default='gated')
        parser.add('--random_background', action='store_bool', default=False)  
    
    def create(self, args):
        # returns dictionary with pipeline components
        
        num_input_channels = args.descriptor_size
        if not args.ray_block_args['use_alpha']:
            num_input_channels = num_input_channels-1

        input_channels = [num_input_channels] * args.num_mipmap

        net = UNet(
            num_input_channels=input_channels, 
            num_output_channels=args.num_output_channels, 
            feature_scale=4, 
            more_layers=0, 
            upsample_mode='bilinear', 
            norm_layer='bn', 
            last_act='',  
            conv_block=args.conv_block
            )
            
        scenes = {}
        textures = {}
        
        if args.inference:
            args.dataset_names = [args.dataset_name] 
            args.crop_size = args.image_size  
        
        for name in  args.dataset_names:
            scene, texture = self._make_scene_and_texture(args, name)
            scenes[name] = scene
            textures[name] = texture
  
        dataset_factory = ImageDatasetFactory() 
        datasets = dataset_factory.get_datasets(args)

        if args.inference:
            self.ds_test = datasets['test']
        else:
            self.ds_train, self.ds_val = datasets['train'], datasets['val']
        # print('Renderer')
        renderer = self._make_renderer(args)        
        # print('Ray block') 
        ray_block = args.ray_block_module(in_features=num_input_channels, **args.ray_block_args)           
            
        self.optimizer = None
        self._extra_optimizer = None
        
        if not args.inference: 
            if len(textures) == 1:
                self._extra_optimizer = TextureOptimizerClass(textures[name].parameters(), lr=args.texture_lr)

            if net is not None:
                self.optimizer = optim.Adam(net.parameters(), lr=args.lr)
                
            self.criterion = args.criterion_module(**args.criterion_args).cuda()
        
             

        self.args = args
        # print('Model')
        self.model = args.pipeline_model_module(ray_block=ray_block, net=net, renderer=renderer, 
                                                textures=textures, scenes=scenes, 
                                                crop_size=args.crop_size,  
                                                ordered_keys=args.dataset_names, 
                                                output_dim=ray_block.output_dim,
                                                num_output_channels=args.num_output_channels)
    
    def extra_optimizer(self, dataset):
        lr_drop = 1 if self.optimizer is None else self.optimizer.param_groups[0]['lr'] / self.args.lr

        print('lr_drop', lr_drop)
        
        # if we have single dataset, don't recreate optimizer
        if self._extra_optimizer is not None:
            self._extra_optimizer.param_groups[0]['lr'] = self.args.texture_lr
            return self._extra_optimizer

        param_group = []
        id_module_map = self.model.textures.id_module_map
        for ds in dataset:
            param_group.append(
                {'params': self.model.textures.submodules[id_module_map[ds.id]].parameters()}
            )        

        return TextureOptimizerClass(param_group, lr=self.args.texture_lr * lr_drop)
    
    def dataset_load(self, dataset):
        ds_ids = [ds.id for ds in dataset]
        self.model.dataset_load(ds_ids)

    def dataset_unload(self, *args, **kwargs):
        self.model.dataset_unload()

    def get_net(self):
        return self.model.net
        
    def _make_scene_data(self, scene_config, additional_fields=[]):
        pointcloud = read_pc(scene_config['pointcloud'], additional_fields=additional_fields) 
        
        if 'intrinsic_matrix' in scene_config:
            if scene_config['intrinsic_matrix'].endswith('xml'):
                intrinsic_matrix, (width, height) = intrinsics_from_xml(scene_config['intrinsic_matrix'])
                assert tuple(scene_config['viewport_size']) == (width, height), f'calibration width, height: ({width}, {height})'
            else:
                intrinsic_matrix = np.loadtxt(scene_config['intrinsic_matrix'])[:3, :3]
        else:
            intrinsic_matrix = None

        return {
                'pointcloud': pointcloud,
                'intrinsic_matrix': intrinsic_matrix,
                'viewport_size': scene_config['viewport_size']
                }
    
    def _make_renderer(self, args):
        scales = [2**i for i in range(args.num_mipmap)]  
        projector = args.projector_module()
        OW, OH = args.crop_size

        sampler = args.grid_sampler_module(OW=OW, OH=OH, max_ray_length=args.max_ray_length, scales=scales)
        return args.renderer_module(sampler=sampler, projector=projector, **args.renderer_args) 
    
    def _make_scene_and_texture(self, args, ds_name):
        
        paths_config = load_yaml(args.paths_file, eval_data=True)

        dataset_config = get_dataset_config(paths_config, ds_name)

        scene_config = load_yaml(dataset_config['scene_path'], eval_data=False)
        
        for k in scene_config:
            if isinstance(scene_config[k], str):
                scene_config[k] = fix_relative_path(scene_config[k], dataset_config['scene_path'])

        additional_fields = args.additional_fields if hasattr(args, 'additional_fields') else []
        scene_data = self._make_scene_data(scene_config, additional_fields=additional_fields)  
        
        print(ds_name, scene_data['pointcloud']['xyz'].shape)
        
        requires_grad = ds_name not in args.freeze_textures
        
            
        
        
        texture = args.texture_module(num_channels=args.descriptor_size, \
                               size=scene_data['pointcloud']['xyz'].shape[0], \
                               **args.texture_args, requires_grad=requires_grad)
       
        if hasattr(args, 'texture_weights') and ds_name in args.texture_weights['datasets']:
            load_model_checkpoint(args.texture_weights['path'].format(ds_name=ds_name), texture)
                
        scene_module = args.scene_module if hasattr(args, 'scene_module') else XYZScene
        scene = scene_module(scene_data)
        
        return scene, texture
    
    def state_objects(self):
        objs = {}
        if self.model.net is not None:
            objs['net'] = self.model.net
            
        textures = self.model.textures.submodules        
        objs.update(textures)

        return objs
    
