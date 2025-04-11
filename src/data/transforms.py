import numpy as np
from src.data.rle import rle_to_dense
from src.common import vox_to_object_space

class NoneTransform(object):
    """ Does nothing to the pointcloud, to be used instead of None
    """
    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        return data


# Transformms
class RandomLinearTransformAllPoints(object):
    ''' Random linear transformation class.

    It applies random linear transforms to all points in data (point cloud and query points)

    Args:
        stddev (int): standard deviation
    '''

    def __init__(self, stddev):
        self.stddev = stddev

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        if self.stddev > 0:
            random_translation = np.random.randn( 3 ) * self.stddev
            random_translation = random_translation.astype( np.float32 )
            mat = np.random.randn( 9 ) * self.stddev
            mat = mat.reshape( 3, 3 ).astype( np.float32 )
            for i in range( 3 ) : mat[ i ][ i ] += 1

            fields_to_process = [ "points", "inputs" ]
            if "points_metric" in data.keys():
                fields_to_process += [ "points_metric" ]
            if 'volume.vol' in data.keys():
                fields_to_process += ['volume.vol']

            for t in fields_to_process :
                data[ t ] = np.add( data[ t ] @ mat,  random_translation )

        return data

class ScalePointcloud(object):
    ''' Point cloud scaling transformation class.

    It scales the point cloud data to the unit cube.

    Args:
        cfg: config
    '''
    def __init__(self, cfg):
        self.scaling_coef = cfg['data']['scaling_coef']

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = vox_to_object_space(data[None], data['affine'], data['shape'], scaling_coef=self.scaling_coef, norm_only=True)
        data_out[None] = points
        return data_out

class SubsamplePointcloud(object):
    ''' Point cloud subsampling transformation class.

    It subsamples the point cloud data.

    Args:
        N (int): number of points to be subsampled
    '''
    def __init__(self, cfg):
        self.N = cfg['data']['pointcloud_n']

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dict): data dictionary
        '''
        data_out = data.copy()
        points = data[None]

        indices = np.random.randint(points.shape[0], size=self.N)

        data_out[None] = points[indices, :]
        for field in [ "colors", "surf" ]:
            if field in data: data_out[field] = data_out[field][indices]

        return data_out


class SubsamplePoints(object):
    ''' Points processing transformation class.

    From the compressed segmentation map, this class extracts and subsamples the query
    points for both the computation of the loss and of the metric (Dice or IoU).

    Args:
        N (int): number of points to be subsampled (computation of loss)
        N_metric (int): number of points to be subsampled (computation of metric)
        num_classes (int): number of classes
        seed (int): seed used in functions from the random package
        mapping_dict (dict): dictionary thats maps the original class to the new classes.

    '''
    def __init__(self, N, N_metric, num_classes,  scaling_coef, seed = None, mapping_dict = None):
        self.N = N
        self.N_metric = N_metric
        self.seed = seed
        self.num_classes = num_classes
        self.mapping_dict = mapping_dict
        self.scaling_coef = scaling_coef

    def __call__(self, data):
        ''' Calls the transformation.

        Args:
            data (dictionary): data dictionary
        '''
        data_out = data.copy()
        # test whether we are in validation or not
        if 'occ' in data and self.seed : return data

        dense = rle_to_dense( data[ 'points.rle' ] )

        # apply mapping
        # ~~~~~~~~~~~~~~~~~~~
        if self.mapping_dict is not None:
            k = np.array(list(self.mapping_dict.keys()))
            v = np.array(list(self.mapping_dict.values()))

            mapping_ar = np.zeros(k.max()+1,dtype=v.dtype)
            mapping_ar[k] = v
            dense = mapping_ar[dense]

        # set params
        size = dense.shape[ 0 ]
        shape = data[ "points.shape" ]
        img = dense.reshape( shape )
        affine = data[ "points.affine" ]

        # process points
        # ~~~~~~~~~~~~~~~~~~~

        rng = np.random.default_rng(seed=None)
        idx = rng.integers(size=self.N, low = 0, high=size)
        occ = np.eye(self.num_classes)[dense[ idx ] ]
        xyz = np.transpose( np.array( np.unravel_index(idx, shape ) ) )
        xyz = vox_to_object_space(xyz, affine, shape, scaling_coef=self.scaling_coef) # object in [-0.5,0.5]

        data_out['points'] = xyz
        data_out['points.occ'] = occ

        if 'points_metric.rle' in data.keys():
            rng = np.random.default_rng(seed=self.seed)
            idx = rng.integers(size=self.N_metric, low = 0, high=size)
            occ = np.eye(self.num_classes)[dense[ idx ] ]
            xyz = np.transpose( np.array( np.unravel_index(idx, shape ) ) )
            xyz = vox_to_object_space(xyz, affine, shape, scaling_coef=self.scaling_coef) # object in [-0.5,0.5]
            data_out['points_metric'] = xyz
            data_out['points_metric.occ'] = occ

        return data_out
