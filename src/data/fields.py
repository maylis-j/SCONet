import os
import numpy as np
from src.data.core import Field

class IndexField(Field):
    ''' Basic index field.'''
    def load(self, info):
        ''' Loads the index field.

        Args:
            info (dict): dict with config information on the model
        '''
        return info['idx']


# 3D Fields
class PointsField(Field):
    ''' Point Field.

    It provides the field to load point data. This is used for the query points
    randomly sampled in the object space.

    Args:
        file_name (str): file name
        transform (list): list of transformations which will be applied to the points tensor

    '''
    def __init__(self, file_name, transform=None):
        self.file_name = file_name
        self.transform = transform

    def load(self, info):
        ''' Loads the data point.

        Args:
            info (dict): dict with config information on the model
        '''
        file_path = os.path.join(info['subject_path'], self.file_name)
        points_dict = np.load(file_path)

        data = {
            'rle' : points_dict[ "rle" ].astype( np.int16 ),
            'affine' : points_dict[ "affine" ],
            'shape' : points_dict[ "shape" ]
        }

        if self.transform is not None:
            data = self.transform(data)
        return data


class PointCloudField(Field):
    ''' Point cloud field.

    It provides the field used for point cloud data. These are the points
    randomly sampled on the Canny point cloud.

    Args:
        file_name (str): file name
        transform (list): list of transformations applied to data points
        use_colors (bool): if given, intensity features are used for training
        use_surf (bool): if given, SURF features are used for training
        surf_scale (float): 
    '''
    def __init__(self, file_name, transform=None, use_colors=False, use_surf=False):
        self.file_name = file_name
        self.transform = transform
        self.use_colors = use_colors
        self.use_surf = use_surf

    def load(self, info):
        ''' Loads the data point.

        Args:
            info (dict): dict with config information on the model
        '''
        file_path = os.path.join(info['subject_path'], self.file_name)
        pointcloud_dict = np.load(file_path)
        
        points = pointcloud_dict['points'].astype(np.float32)
        
        data = {
            None: points,
            'affine': info['affine'],
            'shape': info['shape']
        }
        
        if self.use_colors:
            colors = pointcloud_dict['colors'].astype(np.float32)
            data['colors'] = colors
        
        if self.use_surf:
            surf = pointcloud_dict['surf'].astype(np.float32)
            if 'scales' in pointcloud_dict:
                surf = np.multiply(surf, pointcloud_dict['scales'])
            data['surf'] = surf

        if self.transform is not None:
            data = self.transform(data)

        return data
