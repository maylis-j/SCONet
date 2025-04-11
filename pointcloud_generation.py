import os
import subprocess
import tempfile
import argparse
import numpy as np
import pandas as pd
from src import config
from src.tools import load_sitk, get_affine
from src.data.rle import dense_to_rle

homeDir = os.environ['HOME']

### QUERY POINTS FUNCTIONS
##########################

def save_query_points_rle(path, id, id_seg, points_file, row, skip_existing = False):
    out_file = os.path.join(path, 'pointcloud', id, points_file )
    if skip_existing and os.path.exists( out_file ) :
        print( "RLE query points already exist, skipping" )
        return

    file_seg = os.path.join(path, 'segmentations', id_seg + '.nii.gz')
    if not os.path.exists( file_seg ) :
        print( "WARNING!!! Segmentation file does not exist :", file_seg )
        return

    _, array_seg = load_sitk(file_seg)
    array_seg = array_seg.astype( np.uint16 )

    rle = dense_to_rle(array_seg)
    affine = get_affine(row)

    np.savez(out_file,
            rle = rle,
            affine = affine,
            shape = array_seg.shape
    )
    print('Query points successfully saved!')


### POINTCLOUD FUNCTIONS
########################

def normalize_intensity(data):

    data = data.clip(-256,256)
    data = (data+256)/(2*256)
    
    return data

def run_surf3D(exe_path, volume_path, csv_path, surf_extras ):
    command_line=exe_path + " " + volume_path + " -p " + csv_path + " -o " + csv_path[:-4] + " " + surf_extras
    print("Computing SURF3D features: " )
    print(command_line)
    subprocess.run(command_line, shell=True)
    

def save_pointcloud(cfg, path, id, dim_z, exe_path, row,
                    surf_extras = "", resample = None, temp_dir = None, skip_existing = False, 
                    int8 = False):
    '''
    Saves input point cloud (coordinates + normalized intensity + SURF)
    path: data path
    id: id of computed volume
    dim_z: dimension along z axis of computed volume
    exe_path: path of SURF descriptor executable 
    '''
    if skip_existing and os.path.exists( os.path.join( path, 'pointcloud', id, cfg['data']['pointcloud_file']) ) :
        print( "Point cloud already exists, skipping" )
        return 1

    if temp_dir == None:
        temp=tempfile.TemporaryDirectory();
        temp_dir=temp.name

    # Load volumes
    dim = None
    if resample != None : dim = (resample,resample,dim_z)
    affine = get_affine(row,dim=dim)
    ## intensity
    file_img = os.path.join(path, 'volumes', id+'.nii.gz')
    _, img = load_sitk(file_img, dim)    
    img = normalize_intensity(img)
    print('Intensity volume loaded')
    ## canny
    file_canny = os.path.join(path, 'canny', cfg['data']['canny_path'], id + '.nii.gz')
    _, canny = load_sitk(file_canny, dim)
    print('Canny volume loaded')
    
    # Extract coordinates
    xyz = np.argwhere(canny>0)
    colors = img[tuple(xyz.T)]
    
    # Change coordinate space
    rot_mat = affine[:3, :3]
    trans_mat = affine[:3, 3]
    xyz = xyz@rot_mat.T + trans_mat # to object space

    # SURF descriptors computation
    out_path = os.path.join(path,'pointcloud') 
    if not os.path.exists(out_path): os.makedirs(out_path)
    out_path = os.path.join(out_path, id)
    if not os.path.exists(out_path): os.makedirs(out_path)
    df = pd.DataFrame(xyz, columns=['x','y','z'])
    df['scale'] = 4
    csv_path = os.path.join(temp_dir,id +'pointcloud.csv')
    df.to_csv(csv_path, header=False, index=False)
    run_surf3D(exe_path,file_img,csv_path,surf_extras)

    try: 
        surf = np.fromfile(os.path.join(temp_dir,id +'pointcloud.bin'), dtype=np.float32 )
        surf = surf.reshape( xyz.shape[ 0 ], 54 )[:, 6:]
        os.remove(csv_path)
        os.remove(os.path.join(temp_dir,id +'pointcloud.bin'))
        os.remove(os.path.join(temp_dir,id +'pointcloud.json'))
    except : 
        print("Error occured while computing the SURF descriptors! Skipping this subject for now.")
        return 0
    
    # Save
    if int8:
        maxima = np.max( np.abs( surf ) + 1e-6, axis = 0 )
        np.savez(os.path.join(out_path,cfg['data']['pointcloud_file']),
             points=xyz,
             colors=colors,
             surf=np.multiply( surf, 127 / maxima ).astype( np.int8 ),
             scales=maxima / 127)
    else:
        np.savez(os.path.join(out_path,cfg['data']['pointcloud_file']),
             points=xyz, 
             colors=colors,
             surf=surf.astype( np.float16 ))
        
    print('Point cloud successfully saved!')
    return 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Configuration file with details to run code.', required=True)
    parser.add_argument('--id', help='Process only one id')
    parser.add_argument('--exe_path', type=str, help='Path to SURF 3D executable.', required=True)
    parser.add_argument('--temp_dir', help='Custom temporary directory')
    parser.add_argument('-rle', '--query_points_rle', help='write query points (rle)', action = "store_true")
    parser.add_argument('--no_point_cloud', help='Avoid point cloud computation', action = "store_true")
    parser.add_argument('-s', "--skip_existing", help='Skip computation when file already exists', action = "store_true")
    args = parser.parse_args()


    root_path = os.path.dirname(os.path.realpath(__file__))
    cfg = config.load_config(args.config, os.path.join(root_path, 'configs/default.yaml'))

    dataset_path = os.path.join(cfg['data']['data_path'], cfg['data']['dataset'])
    df = pd.read_csv(os.path.join(dataset_path,'info_dataset.csv'))
    
    if cfg['data']['canny_resolution'] > 0 : resample = cfg['data']['canny_resolution']

    for index, row in df.iterrows():
        id = row['id']
        if args.id and args.id != id : continue
        id_seg = row['id_seg']
        print('\n',id)
        
        if not args.no_point_cloud :# ~~ save input point cloud
            save_pointcloud(cfg,
                                    path=dataset_path,
                                    id=id,
                                    dim_z=row['dim_z'],
                                    exe_path=args.exe_path,
                                    surf_extras = cfg['data']['surf_extras'],
                                    resample = resample,
                                    temp_dir=args.temp_dir,
                                    skip_existing = args.skip_existing,
                                    int8 = cfg['data']['int8'],
                                    row=row)

        # # ~~ save query point cloud
        if args.query_points_rle : save_query_points_rle(dataset_path,
                                  id,
                                  id_seg,
                                  cfg['data']['points_file'],
                                  row=row,
                                  skip_existing = args.skip_existing)


    
