from plyfile import PlyData, PlyElement
import torch
import numpy as np


def read_pc(path, additional_fields=[]):
    '''
    additional_fields: ['rgb', 'normals']
    '''
    pc = PlyData.read(path)
    
    vertex = pc['vertex']
    pointcloud = {}
    
    pointcloud['xyz'] = (np.array([vertex[t] for t in ('x', 'y', 'z')])).T

    if 'rgb' in additional_fields:
        pointcloud['rgb'] = np.array([vertex[t] for t in ('red', 'green', 'blue')]).T
        print('vertex', vertex['red'].max())
        if vertex['red'].dtype == np.uint8:
            pointcloud['rgb'] = pointcloud['rgb']/ 255.
        
    if 'normals' in additional_fields:
        pointcloud['normals'] = np.array([vertex[t] for t in ('nx', 'ny', 'nz')]).T
    
    return pointcloud

def write_pc(xyz, rgb, normals, fp):
    n = xyz.shape[0]
    fields = [ ('red', '<u1'), ('green', '<u1'), ('blue', '<u1'), ('x', '<f4'), ('y', '<f4'), ('z', '<f4') , ('nx', '<f4'), ('ny', '<f4'), ('nz', '<f4')]

    vertex_all = np.empty(n, fields)

    vertex_all['x'] = xyz[:,0]
    vertex_all['y'] = xyz[:,1]
    vertex_all['z'] = xyz[:,2]
    
    vertex_all['red'] =  rgb[:,0]
    vertex_all['green'] = rgb[:,1]
    vertex_all['blue'] = rgb[:,2]
    
    vertex_all['nx'] = normals[:,0]
    vertex_all['ny'] = normals[:,1]
    vertex_all['nz'] = normals[:,2]

    ply = PlyData([PlyElement.describe(vertex_all, 'vertex')], text=False)

    ply.text = False
    ply.byte_order='<'
    ply.write(fp)