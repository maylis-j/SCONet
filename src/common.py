import time
import numpy as np
from pynvml_utils import nvidia_smi
# from pynvml.smi import nvidia_smi
import torch

def make_3d_grid(bb_min, bb_max, shape):
    ''' Makes a 3D grid.

    Args:
        bb_min (tuple): bounding box minimum
        bb_max (tuple): bounding box maximum
        shape (tuple): output shape
    '''

    size = shape[0] * shape[1] * shape[2]

    pxs = torch.linspace(bb_min[0], bb_max[0], shape[0])
    pys = torch.linspace(bb_min[1], bb_max[1], shape[1])
    pzs = torch.linspace(bb_min[2], bb_max[2], shape[2])

    pxs = pxs.view(-1, 1, 1).expand(*shape).contiguous().view(size)
    pys = pys.view(1, -1, 1).expand(*shape).contiguous().view(size)
    pzs = pzs.view(1, 1, -1).expand(*shape).contiguous().view(size)
    p = torch.stack([pxs, pys, pzs], dim=1)

    return p

def normalize_3d_coordinate(p, padding=0.1):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        p (tensor): point
        padding (float): conventional padding parameter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    p_nor = p / (1 + padding + 10e-4) # (-0.5, 0.5)
    p_nor = p_nor + 0.5 # range (0, 1)
    # f there are outliers out of the range
    if p_nor.max() >= 1:
        p_nor[p_nor >= 1] = 1 - 10e-4
    if p_nor.min() < 0:
        p_nor[p_nor < 0] = 0.0
    return p_nor


def coordinate2index(x, reso, coord_type='2d'):
    ''' Normalize coordinate to [0, 1] for unit cube experiments.
        Corresponds to our 3D model

    Args:
        x (tensor): coordinate
        reso (int): defined resolution
        coord_type (str): coordinate type
    '''
    x = (x * reso).long()
    if coord_type == '2d': # plane
        index = x[:, :, 0] + reso * x[:, :, 1]
    elif coord_type == '3d': # grid
        if len(x.shape)==3:
            index = x[:, :, 0] + reso * (x[:, :, 1] + reso * x[:, :, 2])
        else:
            index = x[:, 0] + reso * (x[:, 1] + reso * x[ :, 2])
    if len(x.shape)==3:
        index = index[:, None, :]
    else:
        index = index[None, :]
    return index


def daemon_process(time_interval, out_path):
    while True:
        nvsmi = nvidia_smi.getInstance()
        dictm = nvsmi.DeviceQuery("memory.free, memory.total")
        gpu_memory = dictm['gpu'][0]['fb_memory_usage']['total'] - dictm['gpu'][0]['fb_memory_usage']['free']

        with open(out_path, 'a')as f:
            f.write(str(gpu_memory)+"\n")

        time.sleep(time_interval)

def get_inputs(data):
    '''
    Load inputs. Concatenates features if necessary.
    '''
    inputs = data.get('inputs', torch.empty(1, 0))
    if 'inputs.colors' in data:
        inputs_colors = data.get('inputs.colors') # B x N
        inputs_colors = torch.reshape(inputs_colors, (inputs.shape[0], inputs.shape[1], 1)) # B x N x 1
        inputs = torch.cat((inputs,inputs_colors), dim=2)
    if 'inputs.surf' in data:
        inputs_surf = data.get('inputs.surf') # B x N x 48
        inputs = torch.cat((inputs,inputs_surf), dim=2)
    return inputs

def vox_to_object_space(xyz, affine, dim, scaling_coef=400, norm_only=False):
    '''
    Transposes the coordinates from the voxelized space to the object space in [-0.5,0.5].

    Args:
        xyz (array): coordinates to transpose
        affine (array): affine array of related volume
        dim (list): shape of related volume
        scaling_coef (int): norrmalization coefficient to fit in unit cube 
        norm_only (bool): only normalize when coordinates are already in the object space
    '''
    # To unnormalized object space
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if not norm_only :
        rot_mat = affine[:3, :3]
        trans_mat = affine[:3, 3]
        xyz = xyz@rot_mat.T + trans_mat

    # To normalized object space
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~
    # shift
    centers = []
    for i, d in enumerate(dim):
        c = (2*affine[i][3] + d*affine[i][i])/2
        centers += [c]
    xyz = np.subtract(xyz, np.array(centers).astype('float32'))

    # normalization
    scale=np.array([1/scaling_coef,1/scaling_coef,1/scaling_coef]).astype('float32')
    xyz= np.multiply(xyz,scale)

    return xyz


def object_to_vox_space(xyz, affine, dim, scaling_coef=400):
    '''
    Transposes the coordinates from the object space in [-0.5,0.5] to the voxelized space.
    '''
    # To unnormalized object space
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # normalization
    scale=np.array([scaling_coef,scaling_coef,scaling_coef]).astype('float32')
    xyz= np.multiply(xyz,scale)

    # shift
    centers = []
    for i, d in enumerate(dim):
        c = (2*affine[i][3] + d*affine[i][i])/2
        centers += [c]
    xyz = np.add(xyz, np.array(centers).astype('float32'))

    # To voxel space
    # ~~~~~~~~~~~~~~
    rot_mat = affine[:3, :3]
    trans_mat = affine[:3, 3]
    xyz = ((xyz - trans_mat)@np.linalg.inv(rot_mat))

    return xyz