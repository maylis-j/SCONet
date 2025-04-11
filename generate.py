import os
from multiprocessing import Process
import time

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src import config
from src.checkpoints import CheckpointIO
from src.utils.io import export_pointcloud
from src.common import daemon_process
from src.tools import get_affine, to_nii

parser = argparse.ArgumentParser(
    description='Extract segmented volumes from occupancy process.'
)
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('--no-cuda', action='store_true', help='Do not use cuda.')

args = parser.parse_args()
root_path = os.path.dirname(os.path.realpath(__file__))
cfg = config.load_config(args.config, os.path.join(root_path, 'configs/default.yaml'))
is_cuda = (torch.cuda.is_available() and not args.no_cuda)
device = torch.device("cuda" if is_cuda else "cpu")
device_name = torch.cuda.get_device_name(device=device)

seed = cfg['generation']['generation_seed']
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

out_dir = cfg['training']['out_dir']
generation_dir = os.path.join(out_dir, cfg['generation']['generation_dir'])
out_time_file = os.path.join(generation_dir, 'perf_generation_full.csv')

# Dataset
dataset = config.get_dataset('test', cfg, return_idx=True)

# Model
model = config.get_model(cfg, device=device, dataset=dataset)

checkpoint_io = CheckpointIO(out_dir, model=model, device=device)
checkpoint_io.load(cfg['generation']['model_file'])

# Generator
generator = config.get_generator(model, cfg, device=device)

# Determine what to generate
generate_volume = cfg['generation']['generate_volume']
copy_input = cfg['generation']['copy_input']

# Loader
test_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, num_workers=8, shuffle=False)

# Statistics
perf_dicts = []

# Generate
model.eval()

df_info = pd.read_csv(os.path.join(cfg['data']['data_path'], cfg['data']['dataset'], 'info_dataset.csv'))

# Check GPU usage during segmentation generation
gpu_usage_path = os.path.join(generation_dir, 'gpu_usage_inference.txt')
if not os.path.exists(generation_dir):
        os.makedirs(generation_dir)
if device != 'cpu':
    device_name = torch.cuda.get_device_name(device=device)
    with open(gpu_usage_path, 'w')as f:
        f.write(device_name + "\n")
    p1 = Process(target=daemon_process, args=(1, gpu_usage_path))
    p1.daemon = True
    p1.start()

# Output folders
volume_dir = os.path.join(generation_dir, 'volume')
in_dir = os.path.join(generation_dir, 'input')
if generate_volume and not os.path.exists(volume_dir):
    os.makedirs(volume_dir)

if copy_input and not os.path.exists(in_dir):
    os.makedirs(in_dir)

for it, data in enumerate(tqdm(test_loader)):

    # Get index etc.
    idx = data['idx'].item()

    try:
        subject_dict = dataset.get_subject_dict(idx)
    except AttributeError:
        subject_dict = {'subject': str(idx), 'category': 'n/a'}
    subjectname = subject_dict['subject']
    
    # Timing dict
    perf_dict = {
        'idx': idx,
        'subjectname': subjectname,
    }

    # Generate outputs
    out_file_dict = {}

    row = df_info[df_info['id']==subjectname]
    
    if cfg['generation']['low_res']:
        row['affine_0'].values[0] = row['affine_0'].values[0]*2
        row['affine_5'].values[0] = row['affine_5'].values[0]*2
        row['dim_x'].values[0] = row['dim_x'].values[0] // 2
        row['dim_y'].values[0] = row['dim_y'].values[0] // 2

    spacing = [row['affine_0'].values[0], row['affine_5'].values[0], row['affine_10'].values[0]]
    dim = [row['dim_x'].values[0], row['dim_y'].values[0], row['dim_z'].values[0]]

    if generate_volume:
        t0 = time.time()
        affine = get_affine(row)
        out = generator.generate_volume(data, 
                                            spacing=spacing, 
                                            dim=dim)
        
        perf_dict['inference_time'] = time.time() - t0

        volume = out['volume']

        # Get statistics
        stats_dict = {}
        if 'stats_dict' in out :
            stats_dict = out['stats_dict']

        perf_dict.update(stats_dict)

        # Write output
        volume_out_file = os.path.join(volume_dir, '%s.nii.gz' % subjectname)
        to_nii(volume.astype(np.float32),volume_out_file,affine)

        if 'flop_counter' in out :
            perf_dict['flops'] = out['flop_counter']


    if cfg['generation']['copy_input']:
        # Save inputs
        inputs_path = os.path.join(in_dir, '%s.ply' % subjectname)
        inputs = data['inputs'].squeeze(0).cpu().numpy()
        export_pointcloud(inputs, inputs_path, False)
        out_file_dict['in'] = inputs_path

    # save GPU and time usage
    perf_dicts.append(perf_dict)

# Create pandas dataframe and save
time_df = pd.DataFrame(perf_dicts)
time_df.set_index(['idx'], inplace=True)
time_df.to_csv(out_time_file)
