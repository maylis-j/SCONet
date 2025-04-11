import csv
import os
import argparse
import nibabel as nib
from src import config

def save_stats_nii(img_dir, seg_dir, csv_path, to_RAS=False):
    '''
    Saves statistics from the nifti files to a csv file:
        - dimensions along x, y z
        - spacing along x, y, z
        - origin in x, y, z
        - complete affine matrix
    Args:
        img_dir (str): directory that contains the CT/MRI volumes
        seg_dir (str): directory that contains the corresponding 
        segmentation maps
        csv_path (str): path to the CSV file where the stats must be saved
        to_RAS (bool): if True, non-RAS-oriented volumes and segmentation maps 
        are changed to RAS
    '''
    with open(csv_path, 'w') as file:
        writer = csv.writer(file)
        writer.writerow(["id", "id_seg",
                         "dim_x", "dim_y", "dim_z", 
                         "spacing_x", "spacing_y", "spacing_z", 
                         "origin_x", "origin_y", "origin_z",
                         "orientation"]+["affine_"+str(i) for i in range(16)])
        file.close()

    with open(os.path.join(img_dir,'id.lst'), 'r') as f:
        list_id = f.read().split('\n')
        list_id = list(filter(lambda x: len(x) > 0, list_id))
        f.close()
    with open(os.path.join(seg_dir,'id_seg.lst'), 'r') as f:
        list_id_seg = f.read().split('\n')
        list_id_seg = list(filter(lambda x: len(x) > 0, list_id_seg))
        f.close()

    for i,id in enumerate(list_id):
        print(id)
        id_seg = list_id_seg[i]

        img = nib.load(os.path.join(img_dir,id+'.nii.gz'))

        # in case orientation is not RAS
        orientation = nib.aff2axcodes(img.affine)
        orientation = orientation[0] + orientation[1] + orientation[2]
        if to_RAS and orientation != 'RAS':
            img=nib.as_closest_canonical(img)
            img_data = img.get_fdata()
            img_conv = nib.Nifti1Image(
                img_data.astype(img.header.get_data_dtype()), 
                img.affine, img.header)
            nib.save(img_conv, os.path.join(img_dir,id+'_RAS.nii.gz'))
            print(id, " transformed to RAS in ", id + '_RAS')
            
            seg = nib.load(os.path.join(seg_dir,id_seg+'.nii.gz'))
            seg = nib.as_closest_canonical(seg)
            seg_data = seg.get_fdata()
            seg_conv = nib.Nifti1Image(
                seg_data.astype(seg.header.get_data_dtype()), 
                seg.affine, seg.header)
            nib.save(seg_conv, os.path.join(seg_dir,id_seg+'_RAS.nii.gz'))
            id = id + '_RAS'
            id_seg = id_seg + '_RAS'
            orientation = 'RAS'
        
        # save stats
        with open(csv_path, 'a') as file:
            writer = csv.writer(file)
            row = [id] + [id_seg]
            row += list(img.header['dim'][1:4])
            row += list(img.header['pixdim'][1:4])
            row += [img.header['qoffset_x']]
            row += [img.header['qoffset_y']]
            row += [img.header['qoffset_z']]
            row += [orientation] 
            row += img.affine.flatten().tolist()
            writer.writerow(row)
            file.close()


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration file with details to run code', required=True)
    args = parser.parse_args()

    root_path = os.path.dirname(os.path.realpath(__file__))
    cfg = config.load_config(args.config, os.path.join(root_path, 'configs/default.yaml'))

    dataset_path = os.path.join(cfg['data']['data_path'], cfg['data']['dataset'])
    img_dir = os.path.join(dataset_path,'volumes')
    seg_dir = os.path.join(dataset_path,'segmentations')
    csv_path = os.path.join(dataset_path,'info_dataset.csv')
    
    save_stats_nii(img_dir=img_dir,
                   seg_dir=seg_dir,
                   csv_path=csv_path,
                   to_RAS=cfg['data']['to_RAS'])
