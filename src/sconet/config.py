# import os
from torch import nn
from src.encoder import encoder_dict
from src.sconet import models, training
from src.sconet import generation
from src import data
from src.losses import _create_loss
# from torchvision import transforms


def get_model(cfg, device=None, dataset=None, **kwargs):
    ''' Return the Occupancy Network model.

    Args:
        cfg (dict): imported yaml config 
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    decoder = cfg['model']['decoder']
    encoder = cfg['model']['encoder']
    dim = cfg['data']['dim']
    c_dim = cfg['model']['c_dim'] # latent size
    decoder_kwargs = cfg['model']['decoder_kwargs']
    encoder_kwargs = cfg['model']['encoder_kwargs']
    padding = cfg['data']['padding']
  
    decoder = models.decoder_dict[decoder](
        dim=dim, c_dim=c_dim, padding=padding,
        **decoder_kwargs
    )

    if encoder == 'idx':
        encoder = nn.Embedding(len(dataset), c_dim)
    elif encoder is not None:
        encoder = encoder_dict[encoder](
            dim=dim, c_dim=c_dim, padding=padding,
            **encoder_kwargs
        )
    else:
        encoder = None
    
    model = models.ConvolutionalOccupancyNetwork(
        decoder, encoder, device=device,
    )

    return model


def get_trainer(model, optimizer, cfg, device, **kwargs):
    ''' Returns the trainer object.

    Args:
        model (nn.Module): the Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
    threshold = cfg['generation']['threshold']
    metric = cfg['training']['model_selection_metric']
    loss = _create_loss(cfg['training']['loss'], 
                        cfg['training']['loss_parameters'])

    compute_metric_on_training_set = cfg["training"]["compute_metric_on_training_set"]
    trainer = training.Trainer(
        model, optimizer,
        device=device, threshold=threshold,
        metric=metric,
        loss=loss,
        compute_metric_on_training_set=compute_metric_on_training_set,
    )

    return trainer


def get_generator(model, cfg, device, **kwargs):
    ''' Returns the generator object.

    Args:
        model (nn.Module): Occupancy Network model
        cfg (dict): imported yaml config
        device (device): pytorch device
    '''
        
    generator = generation.Generator3D(
        model,
        device=device,
        cfg=cfg,
    )
    return generator


def get_data_fields(mode, cfg):
    ''' Returns the data fields.

    Args:
        mode (str): the mode which is used
        cfg (dict): imported yaml config
    '''
    points_transform = data.NoneTransform()
    fields = {}
    if cfg['data']['points_file'] is not None:
        fields['points'] = data.PointsField(
                    cfg['data']['points_file'], points_transform
                    )

    if mode == 'train' and not cfg[ "training" ][ "compute_metric_on_training_set" ]:
        return fields

    if mode in ('train','val', 'test'): 
        points_metric_file = cfg['data']['points_metric_file']
        points_transform = data.NoneTransform()
        if points_metric_file is not None:
            fields['points_metric'] = data.PointsField(
                    points_metric_file, points_transform
                    )

    return fields
