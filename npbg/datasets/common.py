import os, sys
import yaml

from functools import lru_cache

import numpy as np
import cv2

import torch

RGBA = 4
RGB = 3

def write_image(image, path):
    image = np.clip(image*255, 0, 255).astype(np.uint8)

    if image.shape[-1]==RGBA:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    elif image.shape[-1]==RGB:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(path, image)

@lru_cache(maxsize = 1000)
def load_image(path, *args):
    img = cv2.imread(path, *args)

    if img.shape[-1]==RGBA:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
    elif img.shape[-1]==RGB:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    assert img is not None, f'could not load {path}'
    return img.copy()


def any2float(img):
    if isinstance(img, np.ndarray):
        out = img.astype(np.float32)
        if img.dtype == np.uint16:
            out /= 65535
        elif img.dtype == np.uint8:
            out /= 255
    elif torch.is_tensor(img):
        out = img.float()
        if img.dtype == torch.int16:
            out /= 65535
        elif img.dtype == torch.uint8:
            out /= 255
    else:
        raise TypeError('img must be numpy array or torch tensor')

    return out


def rescale_K(K_, sx, sy, keep_fov=True):
    K = K_.copy()
    K[0, 2] = sx * K[0, 2]
    K[1, 2] = sy * K[1, 2]
    if keep_fov:
        K[0, 0] = sx * K[0, 0]
        K[1, 1] = sy * K[1, 1]
    return K

def fit_size(x, d=16):
    return x[..., :d*(x.shape[-2]//d), :d*(x.shape[-1]//d)]


class ToTensor(object):
    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        img = any2float(img)
        
        return img.permute(2, 0, 1).contiguous()

    def __repr__(self):
        return self.__class__.__name__ + '()'


def get_dataset_config(yml, dataset):
    join = os.path.join 

    try:
        myhost = os.uname()[1]
    except:
        myhost = ''
    if myhost in yml:
        data_root = yml[myhost]['data_root'] if 'data_root' in yml[myhost] else ''
    else:
        data_root = yml['data_root'] if 'data_root' in yml else '/'

    ds = yml['datasets'][dataset]

    for k in ds:
        if 'path' in k:
            ds[k] = join(data_root, ds[k])

    return ds


def split_lists(config, lists):
    sz = [len(l) for l in lists]
    assert len(set(sz)) == 1, f'list sizes differ {sz}'

    splits = []
    train_inds, val_inds = [], []
    if 'train_ratio' in config:
        train_ratio = float(config['train_ratio'])
        train_n = int(sz[0] * train_ratio)
        train_inds, val_inds = np.split(np.random.permutation(sz[0]), [train_n])
    else:
        val_step = int(config['val_step'])
        train_drop = int(config['train_drop'])
        for i in range(sz[0]):
            if i % val_step == 0:
                val_inds.append(i)
            elif train_drop < i % val_step < val_step - train_drop:
                train_inds.append(i)

    # print(train_inds)
    # print( val_inds)

    for lst in lists:
        lst = np.array(lst)
        splits.append([lst[train_inds], lst[val_inds]])
        # print('--')
        # [print(x) for x in lst[train_inds]]
        # print('-')
        # [print(x) for x in lst[val_inds]]


    return splits