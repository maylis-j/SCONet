from fvcore.nn import FlopCountAnalysis, flop_count_table
import numpy as np
import time
import torch
from src.common import make_3d_grid
from src.jit_handles import _SUPPORTED_OPS

class Generator3D(object):
    '''  Generator class for Occupancy Networks.

    It provides functions to generate the final mesh as well refining options.

    Args:
        model (nn.Module): trained Occupancy Network model
        points_batch_size (int): batch size for points evaluation
        threshold (float): threshold value
        device (device): pytorch device
        cfg (dict): imported yaml config
    '''

    def __init__(self, model, points_batch_size=100000,
                 device=None, cfg={}):
        self.model = model.to(device)
        self.points_batch_size = points_batch_size
        self.device = device
        self.cfg = cfg
        self.threshold = cfg['generation']['threshold']
        self.scaling_coef = cfg['data']['scaling_coef']
        self.num_classes = cfg['model']['decoder_kwargs']['num_classes']
        self.compute_flops = cfg['generation']['compute_flops']


    def generate_volume(self, data, spacing, dim, return_stats=True):
        ''' Generates the output mesh.

        Args:
            data (tensor): data tensor
            return_stats (bool): whether stats should be returned
        '''
        self.model.eval()
        device = self.device
        stats_dict = {}
        kwargs = {}

        inputs = data.get('inputs', torch.empty(1, 0))
        if 'inputs.colors' in data:
            inputs_colors = data.get('inputs.colors') # B x N
            inputs_colors = torch.reshape(inputs_colors, (inputs.shape[0], inputs.shape[1], 1)) # B x N x 1
            inputs = torch.cat((inputs,inputs_colors), dim=2)
        if 'inputs.surf' in data:
            inputs_surf = data.get('inputs.surf') # B x N x 48
            inputs = torch.cat((inputs,inputs_surf), dim=2)        
        inputs = inputs.to(device)
        
        t0 = time.time()
        with torch.inference_mode():
            c = self.model.encode_inputs(inputs)

            if self.compute_flops : 
                flops = FlopCountAnalysis(self.model.encoder,(inputs,)).uncalled_modules_warnings(False).unsupported_ops_warnings(False).set_op_handle(**_SUPPORTED_OPS)
                self.flop_counter = flops.total()

        stats_dict['time (encode inputs)'] = time.time() - t0
  
        volume = self.generate_volume_from_latent(c=c, 
                                                  spacing=spacing, 
                                                  dim=dim, 
                                                  **kwargs)

        out = {}
        out['volume'] = volume
        out['stats_dict'] = stats_dict
        if self.compute_flops:
            out['flop_counter'] = self.flop_counter

        return out
    
    
    def generate_volume_from_latent(self, spacing, dim, c=None, **kwargs):
        ''' Generates mesh from latent.
            Works for shapes normalized to a unit cube

        Args:
            c (tensor): latent conditioned code c
            stats_dict (dict): stats dictionary
        '''

        bb_min = [-spacing[i]*dim[i]/(2*self.scaling_coef) for i in range(3)] # corresponds to voxels (0,0,0)
        bb_max = [spacing[i]*dim[i]/(2*self.scaling_coef) for i in range(3)]

        pointsf = make_3d_grid(
            bb_min, bb_max, dim
        )        

        values = self.eval_points(pointsf, c, **kwargs).cpu().numpy()

        value_grid = values.reshape(dim[0],dim[1],dim[2],self.num_classes)

        return value_grid
    

    def eval_points(self, p, c=None, **kwargs):
        ''' Evaluates the occupancy values for the points.

        Args:
            p (tensor): points 
            c (tensor): encoded feature volumes
        '''
        p_split = torch.split(p, self.points_batch_size)
        occ_hats = []
        for pi in p_split:
            pi = pi.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                occ_hat = self.model.decode(pi, c, **kwargs)
            occ_hats.append(occ_hat.squeeze(0).detach().cpu())

            if self.compute_flops : 
                flops = FlopCountAnalysis(self.model.decoder, (pi,c)).uncalled_modules_warnings(False).set_op_handle(**_SUPPORTED_OPS)
                self.flop_counter += flops.total()
        
        occ_hat = torch.cat(occ_hats, dim=0)
        return occ_hat