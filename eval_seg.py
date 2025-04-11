import os
import pandas as pd
import SimpleITK as sitk
import argparse
import json

from src import config
from src.tools import resample_image, change_label_nii
from src.metrics import compute_hausdorff, compute_dice_per_organ

def eval_all_multi(gt_dir, pred_dir, lst_id_path, df_info, classes, mapping_file=None):
    '''
    Saves the evaluation metrics in CSV file.

    Args:
        gt_dir (str): ground truth directory
        pred_dir (str): predicton directory
        lst_id_path (str): path to list of testing split
        df_info (pd.Dataframe): dataframe with dataset information
        classes (list): names of all the classes to process
        mapping_file (str): path to mapping json file
    '''
    mapping = {'use_mapping': False}
    if mapping_file is not None:
        with open(mapping_file, 'r') as f:
            mapping['mapping'] = json.load(f)
        mapping['use_mapping'] = True

    # Compute metrics for all subjects
    metrics = []
    f = open(lst_id_path, "r")
    for i in f:
        info = i.split('\n')[0]
        id = info.split(',')[0]
        row = df_info[df_info['id']==id]
        id_seg = row['id_seg'].values[0]
        
        gt_path = os.path.join(gt_dir, id_seg + '.nii.gz')
        pred_path = os.path.join(pred_dir, id + '.nii.gz')

        subject_metrics = {}
        subject_metrics['id'] = id
        subject_metrics.update(eval_seg(gt_path,pred_path,classes,mapping))
        metrics += [subject_metrics]
        print("Subject ", subject_metrics['id'], 
              " evaluated with dice=", subject_metrics['dice_tot'],
              " and hd =", subject_metrics['hd_tot'])
    df_metrics = pd.DataFrame(metrics)

    # Average for each metric
    columns = df_metrics.columns
    avg = ['Avg']
    std = ['Std']
    for col in columns[1:]:
        avg += [df_metrics.loc[:, col].mean()]
        std += [df_metrics.loc[:, col].std()]
    df_metrics.loc[len(df_metrics)] = avg
    df_metrics.loc[len(df_metrics)] = std

    # Arrange columns
    columns_all = ['id'] + [x for x in columns if "dice" in x] + [x for x in columns if "hd" in x]
    df_metrics_all = df_metrics[columns_all]
    df_metrics_all.to_csv(os.path.join(pred_dir,'eval_all.csv'))

    # Save
    columns = ['id'] + [x for x in columns if "tot" in x]
    df_metrics = df_metrics[columns]
    df_metrics.to_csv(os.path.join(pred_dir, 'eval.csv'))


def eval_seg(gt_path, pred_path, classes, mapping):
    '''
    Computes evaluation metrics for one subject.
    
    Args :
        gt_path (str): ground truth path
        pred_path (str): predicton path
        classes (list): names of all the classes to process
        mapping (dict): mapping dictionnary
    '''
    gt = sitk.ReadImage(gt_path, outputPixelType=sitk.sitkUInt8)
    pred = sitk.ReadImage(pred_path, outputPixelType=sitk.sitkUInt8)

    if pred.GetSize() != gt.GetSize() or pred.GetSpacing() != gt.GetSpacing() :
        print('Resampling predicted segmentation')
        pred = resample_image(pred,
                              gt.GetSize(), 
                              out_spacing=gt.GetSpacing())
    
    if mapping['use_mapping']:
        mapping['mapping'] =  {int(k): v for k, v in mapping['mapping'].items()}
        gt = change_label_nii(gt, mapping['mapping'])

    metrics = {}
    metrics.update(compute_hausdorff(gt, pred, classes))

    gt_data = sitk.GetArrayFromImage(gt)
    pred_data = sitk.GetArrayFromImage(pred)
    metrics.update(compute_dice_per_organ(gt_data,pred_data,classes=classes))

    return metrics


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to config file.')
    args = parser.parse_args()

    root_path = os.path.dirname(os.path.realpath(__file__))
    cfg = config.load_config(args.config, os.path.join(root_path, 'configs/default.yaml'))
    
    dataset_path = os.path.join(cfg['data']['data_path'],cfg['data']['dataset'])
    pred_dir = os.path.join(cfg['training']['out_dir'],cfg['generation']['generation_dir'],'volume')
    gt_dir = os.path.join(dataset_path,'segmentations')
    
    mapping_file = None
    if cfg['data']['mapping_file'] is not None:
        mapping_file = os.path.join(dataset_path,cfg['data']['mapping_file'])
        
    df_info = pd.read_csv(os.path.join(dataset_path, 'info_dataset.csv'))
    
    eval_all_multi(gt_dir,
                   pred_dir,
                   lst_id_path=os.path.join(dataset_path, cfg['data']['test_split']+'.lst'), 
                   df_info=df_info,
                   classes=cfg['data']['classes'],
                   mapping_file = mapping_file)
