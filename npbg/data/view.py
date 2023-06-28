import os, sys
import cv2
import numpy as np
import pickle
import time
import yaml

import torch

import xml.etree.ElementTree as ET
# from npbg.data.matrix_torch import (create_rotation_matrix_x, create_rotation_matrix_y, create_rotation_matrix_z, create_translation_matrix)

def from_extrinsic(extrinsic, fix_axes=False):
    if fix_axes:
        extrinsic[:, 1:3] *= -1
    R = extrinsic[:3, :3]
    T = extrinsic[:3, 3]
    view_matrix = np.identity(4)
    view_matrix[:3,:3] = R.T
    view_matrix[:3, 3] = -R.T@T
    return view_matrix

def from_pose(local_rotation, local_position):
    rotation_x = create_rotation_matrix_x(-local_rotation[0])
    rotation_y = create_rotation_matrix_y(-local_rotation[1])
    rotation_z = create_rotation_matrix_z(-local_rotation[2])
    
    translation = create_translation_matrix(-local_position[0], -local_position[1], -local_position[2])
    rotation = torch.mm(rotation_x, rotation_y)
    rotation = torch.mm(rotation, rotation_z)
    transform = torch.mm(rotation, translation)
    
    return transform.t()

def crop_proj_matrix(pm, old_w, old_h, new_w, new_h):
    # NOTE: this is not precise
    old_cx = old_w / 2
    old_cy = old_h / 2
    new_cx = new_w / 2
    new_cy = new_h / 2

    pm_new = pm.copy()
    pm_new[0,0] = pm[0,0]*old_w/new_w
    pm_new[0,2] = (pm[0,2]-1)*old_w*new_cx/old_cx/new_w + 1
    pm_new[1,1] = pm[1,1]*old_h/new_h
    pm_new[1,2] = (pm[0,2]+1)*old_h*new_cy/old_cy/new_h - 1
    return pm_new


def recalc_proj_matrix_planes(pm, new_near=.01, new_far=1000.):
    depth = float(new_far - new_near)
    q = -(new_far + new_near) / depth
    qn = -2 * (new_far * new_near) / depth

    out = pm.copy()

    # Override near and far planes 
    out[2, 2] = q
    out[2, 3] = qn

    return out


def get_proj_matrix(K, image_size, znear=.01, zfar=1000.):
    fx = K[0,0]                                                                                                                                                                           
    fy = K[1,1]
    cx = K[0,2]                                                                                                                                                                           
    cy = K[1,2]
    width, height = image_size
    m = np.zeros((4, 4))
    m[0][0] = 2.0 * fx / width;
    m[0][1] = 0.0;
    m[0][2] = 0.0;
    m[0][3] = 0.0;

    m[1][0] = 0.0;
    m[1][1] = 2.0 * fy / height;
    m[1][2] = 0.0;
    m[1][3] = 0.0;

    m[2][0] = 1.0 - 2.0 * cx / width;
    m[2][1] = 2.0 * cy / height - 1.0;
    m[2][2] = (zfar + znear) / (znear - zfar);
    m[2][3] = -1.0;

    m[3][0] = 0.0;
    m[3][1] = 0.0;
    m[3][2] = 2.0 * zfar * znear / (znear - zfar);
    m[3][3] = 0.0;

    return m.T


def crop_intrinsic_matrix(K, old_size, new_size):
    K = K.copy()
    K[0, 2] = new_size[0] * K[0, 2] / old_size[0]
    K[1, 2] = new_size[1] * K[1, 2] / old_size[1]
    return K


def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width/2
    cy = height/2
    
    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)


def extrinsics_from_xml(xml_file, verbose=False):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)
    
    view_matrices = []
#     labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort

def view_from_txt_extrinsics(path):
    vm = np.loadtxt(path).reshape(-1,4,4)
    vm = [from_extrinsic(view_matrix).T for view_matrix in vm]
    return vm
    
def extrinsics_from_view_matrix(path):
    vm = np.loadtxt(path).reshape(-1,4,4)
    vm, ids = get_valid_matrices(vm)

    return vm, ids