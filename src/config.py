import json
import os
from torchvision import transforms
import yaml
from src import data
from src import sconet
from src.data.transforms import RandomLinearTransformAllPoints

method_dict = {
    'sconet': sconet
}


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.safe_load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


# Models
def get_model(cfg, device=None, dataset=None):
    ''' Returns the model instance.

    Args:
        cfg (dict): config dictionary
        device (device): pytorch device
        dataset (dataset): dataset
    '''
    method = cfg['method']
    model = method_dict[method].config.get_model(
        cfg, device=device, dataset=dataset)
    return model


# Trainer
def get_trainer(model, optimizer, cfg, device):
    ''' Returns a trainer instance.

    Args:
        model (nn.Module): the model which is used
        optimizer (optimizer): pytorch optimizer
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    trainer = method_dict[method].config.get_trainer(
        model, optimizer, cfg, device)
    return trainer


# Generator for final mesh extraction
def get_generator(model, cfg, device):
    ''' Returns a generator instance.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        device (device): pytorch device
    '''
    method = cfg['method']
    generator = method_dict[method].config.get_generator(model, cfg, device)
    return generator


# Datasets
def get_dataset(mode, cfg, return_idx=False):
    ''' Returns the dataset.

    Args:
        model (nn.Module): the model which is used
        cfg (dict): config dictionary
        return_idx (bool): whether to include an ID field
    '''
    method = cfg['method']
    dataset_type = cfg['data']['dataset_type']
    dataset_folder = os.path.join(cfg['data']['data_path'],cfg['data']['dataset'])

    # Get split
    splits = {
        'train': cfg['data']['train_split'],
        'val': cfg['data']['val_split'],
        'test': cfg['data']['test_split'],
    }

    split = splits[mode]
    # Create dataset
    if dataset_type == 'Shapes3D':
        # Dataset fields
        fields={}
        # Method specific fields (usually correspond to output)
        if mode in ['train','val'] :
            fields = method_dict[method].config.get_data_fields(mode, cfg)
        # Input fields
        inputs_field = get_inputs_field(cfg)

        if inputs_field is not None:
            fields['inputs'] = inputs_field

        if return_idx:
            fields['idx'] = data.IndexField()

        transform = data.NoneTransform()

        # Transform after loading all fields
        if mode in ['train','val'] :
            mapping_dict = None
            if cfg['data']['mapping_file'] is not None:
                mapping_path = os.path.join(dataset_folder, cfg['data']['mapping_file'])
                with open(mapping_path, 'r') as f:
                    mapping_dict = json.load(f)
                mapping_dict = {int(k):int(v) for k,v in mapping_dict.items()}
            transform = transforms.Compose([
                transform,
                data.SubsamplePoints(cfg['data']['points_subsample'],
                           cfg['data']['points_subsample_validation'],
                           cfg['model']['decoder_kwargs']['num_classes'],
                           scaling_coef=cfg['data']['scaling_coef'],
                           seed=cfg['data']['points_subsample_validation_seed'],
                           mapping_dict=mapping_dict)])

        if mode == 'train':
            stdev = cfg['data']['random_linear_transform']
        else:
            stdev = 0

        transform = transforms.Compose([
            transform,
            RandomLinearTransformAllPoints( stdev )])
        
        dataset = data.Shapes3dDataset(
            fields,
            split=split,
            cfg=cfg,
            transform=transform
        )

    else:
        raise ValueError('Invalid dataset "%s"' % cfg['data']['dataset_type'])

    return dataset


def get_inputs_field(cfg):
    ''' Returns the inputs fields.

    Args:
        cfg (dict): config dictionary
    '''
    input_type = cfg['data']['input_type']

    if input_type is None:
        inputs_field = None
    elif input_type == 'pointcloud':
        if cfg['data']['pointcloud_n'] is not None and cfg['data']['pointcloud_n'] > 0:
            transform = transforms.Compose([
                data.SubsamplePointcloud(cfg),
                data.ScalePointcloud(cfg)
            ])
        else:
            transform = transforms.Compose([
                data.ScalePointcloud(cfg)
            ])
        inputs_field = data.PointCloudField(
            cfg['data']['pointcloud_file'], transform,
            use_colors = cfg['model']['encoder_kwargs']['use_colors'],
            use_surf = cfg['model']['encoder_kwargs']['use_surf'],
        )
    else:
        raise ValueError(
            'Invalid input type (%s)' % input_type)
    return inputs_field
