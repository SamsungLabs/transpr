import os, sys
from os.path import isdir, isfile, join, exists

from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms

import cv2
import numpy as np

from npbg.gl.utils import (get_proj_matrix, intrinsics_from_xml, 
                       extrinsics_from_xml, extrinsics_from_view_matrix, fix_relative_path)


from npbg.datasets.common import ToTensor, load_image, get_dataset_config, split_lists, rescale_K
from npbg.utils.perform import TicToc
from npbg.data.io.yaml import load_yaml
from npbg.data.io.ply import read_pc
from npbg.data import view
import glob
import pydoc


def rand_(min_, max_, *args):
    return min_ + (max_ - min_) * np.random.rand(*args)



default_target_transform = transforms.Compose([
        ToTensor(),
])


class ImageDataset:
    znear = 0.1
    zfar = 1000
    
    def __init__(self, scene_meta, crop_size,
                 view_list, target_list, mask_list,
                 keep_fov=False, 
                 target_transform=None,
                 num_samples=None, 
                 random_zoom=None, random_shift=None, 
                 random_background=False, backgrounds_list=[],
                 alpha_jitter=False, min_alpha=0.1, max_alpha=1.0, alpha_jitter_p=0.5,
                 random_overlay=False, random_overlay_p = 0.5, bs=None,                  
                 inference=False, **args):

        if isinstance(crop_size, (int, float)):
            crop_size = crop_size, crop_size
        
        # if render image size is different from camera image size, then shift principal point
        K_src = scene_meta['intrinsic_matrix']
        old_size = scene_meta['viewport_size']
        sx = crop_size[0] / old_size[0]
        sy = crop_size[1] / old_size[1]
        K = rescale_K(K_src, sx, sy, keep_fov)
        
        if not inference:
            assert len(view_list) == len(target_list)
        elif random_background:
            assert len(backgrounds_list)==len(view_list)

        self.view_list = [torch.FloatTensor(view) for view in view_list]
        self.target_list = target_list
        self.mask_list = mask_list
        self.crop_size = crop_size
        self.K = K
        self.K_src = K_src
        self.random_zoom = random_zoom
        self.random_shift = random_shift
        self.sx = sx
        self.sy = sy
        self.keep_fov = keep_fov

        # random backgrounds
        self.random_background = random_background
        self.backgrounds_list = backgrounds_list

        crop = transforms.RandomCrop(self.crop_size[::-1], pad_if_needed=True, padding_mode='mirror')

        if inference:
            crop = transforms.CenterCrop(self.crop_size[::-1])

        self.background_transform = transforms.Compose([
                                        transforms.ToPILImage(),
                                        crop,
                                        transforms.ToTensor()])

        self.target_list = target_list
        self.target_transform = default_target_transform if target_transform is None else target_transform
        
        self.num_samples = num_samples if num_samples else len(view_list)
        self.id  = None
        # jitter
        self.alpha_jitter = alpha_jitter
        self.min_alpha = min_alpha
        self.alpha_jitter_p = alpha_jitter_p
        self.max_alpha = max_alpha
        # overlay
        self.random_overlay = random_overlay
        self.random_overlay_p = random_overlay_p
        self.bs = bs
        
        self.inference = inference

        if self.inference:
            self.random_zoom = None
            self.random_shift = None
            self.random_overlay = False
            self.num_samples = len(view_list)

        self.timing = defaultdict(list)
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        idx = idx % len(self.view_list)
        
        tt = TicToc()
        tt.tic()
        
        view_matrix = self.view_list[idx]
        K, proj_matrix, z = self._get_intrinsics()
        
        proj_matrix = torch.FloatTensor(proj_matrix.T)
        
        target = None
        if not self.inference:
            target_filename = self.target_list[idx]
            target = self._get_target_image(target_filename, K)
        
        mask = None            
        if self.mask_list[idx]:
            mask_filename = self.mask_list[idx]
            mask = self._get_target_image(mask_filename, K)
            mask = (mask.sum(2) > 1e-9).astype(np.float32)
            mask = mask[..., None]
            mask = ToTensor()(mask)

        self.timing['get_target'] += [tt.toc()]
        tt.tic()
        
        background = None        
        alpha_jitter = 1.0      
        
        if not self.inference and self.alpha_jitter and (np.random.rand()<self.alpha_jitter_p):
            alpha_jitter = rand_(self.min_alpha, self.max_alpha)
            target = np.clip((target*alpha_jitter), 0, 255)     
        
        if self.random_background:
            background_filename = self.backgrounds_list[idx] if self.inference else np.random.choice(self.backgrounds_list)        
            if background_filename is not None:
                background = load_image(background_filename)
                background = self.background_transform(background.astype(np.uint8))
        
        overlay_index = -1
        if self.random_overlay and (np.random.rand()<self.random_overlay_p):
            overlay_index = np.random.choice(range(self.bs))      
            
        input_ = {
                'view_matrix': view_matrix,
                'proj_matrix': proj_matrix,
                'id': torch.FloatTensor([self.id]),               
        }

        if background is not None:
            input_['background'] = background        
        
        input_['alpha_jitter'] = alpha_jitter   
        input_['overlay_index'] = overlay_index   
                        
        out_dict = { 'input': input_ }
        
        if not self.inference:
            target = self.target_transform(target.astype(np.uint8)) 
            self.timing['transform'] += [tt.toc()] 
            out_dict['target'] = target
            out_dict['target_filename'] = target_filename

        if mask is not None:
            out_dict['mask'] = mask

        return out_dict


    def _get_intrinsics(self, torch_tensor=False):
        K = self.K.copy()
        sx = 1. if self.keep_fov else self.sx
        sy = 1. if self.keep_fov else self.sy
        z = 1.0
        if self.random_zoom:
            z = rand_(*self.random_zoom)
            K[0, 0] *= z
            K[1, 1] *= z
            sx /= z
            sy /= z
        if self.random_shift:
            x, y = rand_(*self.random_shift, 2)
            w = self.crop_size[0] * (1. - sx) / sx / 2.
            h = self.crop_size[1] * (1. - sy) / sy / 2.
            K[0, 2] += x * w
            K[1, 2] += y * h
        proj_matrix = get_proj_matrix(K, self.crop_size, self.znear, self.zfar)
        
        return K, proj_matrix, z
    
    def _get_target_image(self, path, K):
        target = load_image(path, cv2.IMREAD_UNCHANGED)

        H = K @ np.linalg.inv(self.K_src)
        target = cv2.warpPerspective(target, H, tuple(self.crop_size))

        return target


class ImageDatasetFactory:
    def __init__(self):
        pass

    def get_args(self):
        pass

    def get_datasets(self, args):
        datasets_lists = defaultdict(list)
        
        for name in args.dataset_names:
            print(f'creating dataset {name}')
            
            datasets = self._get_datasets(args.paths_file, name, args)

            for k in datasets:                
                datasets[k].id = args.dataset_names.index(name)
                datasets_lists[k].append(datasets[k])

        return datasets_lists
    
    
    def _get_datasets(self, paths_file, ds_name, args):
                
        paths_config = load_yaml(paths_file, eval_data=True)

        dataset_config = get_dataset_config(paths_config, ds_name)
        
        scene_config = load_yaml(dataset_config['scene_path'], eval_data=False)
            
        for k in scene_config:
            if isinstance(scene_config[k], str):
                scene_config[k] = fix_relative_path(scene_config[k], dataset_config['scene_path'])
    
        if scene_config['intrinsic_matrix'].endswith('xml'):
            intrinsic_matrix, (width, height) = intrinsics_from_xml(scene_config['intrinsic_matrix'])
            assert tuple(scene_config['viewport_size']) == (width, height), f'calibration width, height: ({widht}, {height})'
        else:
            intrinsic_matrix = np.loadtxt(scene_config['intrinsic_matrix'])[:3, :3]
            
        scene_meta = { 'intrinsic_matrix': intrinsic_matrix, 'viewport_size': scene_config['viewport_size']}
        if args.inference:
            scene_config['view_matrix'] = args.view_matrix
        if scene_config['view_matrix'].endswith('xml'):
            extrinsics, labels_sort = view.extrinsics_from_xml(scene_config['view_matrix'])
            view_matrix = {label: view.from_extrinsic(extrinsic).T for extrinsic, label in zip(extrinsics, labels_sort)}
        elif scene_config['view_matrix'].endswith('txt'):
            extrinsics = np.loadtxt(scene_config['view_matrix']).reshape((-1,4,4))
            view_matrix = {label: view.from_extrinsic(extrinsic).T for label, extrinsic in enumerate(extrinsics)}
        elif scene_config['view_matrix'].endswith('torch'):
            view_matrix = torch.load(scene_config['view_matrix'])
        
        view_list = [np.array(vm) for vm in view_matrix.values()]
        print('views', len(view_list))
        
        camera_labels = list(view_matrix.keys())
            
        target_list = [join(dataset_config['target_path'], dataset_config['target_name_func'](i)) for i in camera_labels]

        backgrounds_list = []

        args.train_dataset_args['random_background'] = args.random_background

        if args.random_background:
            if args.inference:
                if not hasattr(args, 'background'):
                    backgrounds_list = [None] * len(view_list)
                elif isdir(args.background):
                    assert hasattr(args, 'background_name_func'), f'Missing function for background images names in {args.background}'
                    background_name_func = eval(args.background_name_func)
                    backgrounds_list = [join(args.background, background_name_func(i)) for i in range(len(camera_labels))]

                    for fp in backgrounds_list:
                        assert exists(fp), f'Not found {fp}'

                elif isfile(args.background):
                    backgrounds_list = [args.background]*len(camera_labels)

            elif 'backgrounds_path' in scene_config:                
                bpaths = [join(scene_config['backgrounds_path'], f'*.{img_ext}') for img_ext in ['png', 'jpg', 'jpeg', 'JPG']]

                backgrounds_list = []
                for bpath in bpaths:
                    backgrounds_list+=glob.glob(bpath)   

                assert len(backgrounds_list)>0, 'Missing backgrounds in {}'.format(scene_config['backgrounds_path'])

        if 'mask_path' in scene_config:
            mask_list = [join(dataset_config['mask_path'], dataset_config['mask_name_func'](i)) for i in camera_labels]
        else:
            mask_list = [None] * len(target_list)

        datasets = {}
            
        if args.inference:
            datasets['test'] = ImageDataset(scene_meta=scene_meta, 
                                  view_list=view_list, \
                                  target_list=[], \
                                  mask_list=mask_list,  backgrounds_list=backgrounds_list,\
                                **args.train_dataset_args, crop_size=args.crop_size, inference=args.inference)
            return datasets

        splits = args.splitter_module([view_list, target_list, mask_list], **args.splitter_args) 

        view_list_train, view_list_val = splits[0]
        target_list_train, target_list_val = splits[1]
        mask_list_train, mask_list_val = splits[2]

        if 'random_overlay' in args.train_dataset_args and args.train_dataset_args['random_overlay']:
            args.train_dataset_args['bs'] = args.batch_size

        datasets['train'] = ImageDataset(scene_meta=scene_meta, 
                                  view_list=view_list_train, \
                                  target_list=target_list_train, \
                                  mask_list=mask_list_train,  backgrounds_list=backgrounds_list,\
                                **args.train_dataset_args, crop_size=args.crop_size, inference=args.inference)
        
        datasets['val'] = ImageDataset(scene_meta=scene_meta, \
                                view_list=view_list_val, \
                                target_list=target_list_val, \
                                mask_list=mask_list_val, backgrounds_list=backgrounds_list,\
                               **args.val_dataset_args, crop_size=args.crop_size)

        return datasets