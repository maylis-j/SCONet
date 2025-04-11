import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_max
from src.layers import ResnetBlockFC
from src.common import coordinate2index, normalize_3d_coordinate
from src.encoder.unet3d import UNet3D


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet3d (bool): wether to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        grid_resolution (int): defined resolution for grid feature 
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
        use_colors (bool): if True, intensity features are used
        use_surf (bool): if True, SURF features are used
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                 unet3d=False, unet3d_kwargs=None, 
                 grid_resolution=None, padding=0.1, n_blocks=5,
                 use_colors=False, use_surf=False):
        super().__init__()

        self.pointnet_out_dim = unet3d_kwargs['in_channels']
        self.feature_size = 0
        if use_colors:
            self.feature_size+=1
        if use_surf:
            self.feature_size+=48
        self.unet3d_in_dim = unet3d_kwargs['in_channels'] + self.feature_size

        if unet3d:
            unet3d_kwargs['in_channels'] = self.unet3d_in_dim
            unet3d_kwargs['out_channels'] = c_dim
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_grid = grid_resolution
        self.padding = padding
        self.use_colors = use_colors
        self.use_surf = use_surf

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')
        
        pos_in_c = dim
        
        # Encoder layers
        self.fc_pos = nn.Linear(pos_in_c, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, self.pointnet_out_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim


    def generate_grid_features(self, p, c):
        p_nor = normalize_3d_coordinate(p, padding=self.padding)
        index = coordinate2index(p_nor, self.reso_grid, coord_type='3d')
        # scatter grid features from points
        fea_grid = c.new_zeros(p.size(0), self.unet3d_in_dim, self.reso_grid**3)
        c = c.permute(0, 2, 1)
        fea_grid = scatter_mean(c, index, out=fea_grid) # B x C x reso^3
        fea_grid = fea_grid.reshape(p.size(0), self.unet3d_in_dim, self.reso_grid, self.reso_grid, self.reso_grid) # sparse matrix (B x c_dim x reso x reso x reso)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    def pool_local(self, xy, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        keys = xy.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        # acquire the index for each point
        coord = {}
        index = {}
        coords = p[:,:,:3]

        coord['grid'] = normalize_3d_coordinate(coords, padding=self.padding)
        index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')

        net = self.fc_pos(coords)
        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(coord, index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)
        if self.feature_size>0:
            c = torch.cat((c,torch.reshape(p[:,:,3:],(p.shape[0],p.shape[1],self.feature_size))), dim=2)

        fea = {}
        fea['grid'] = self.generate_grid_features(coords, c)

        return fea
