import os
import time
import argparse
import numpy as np
import pandas as pd
import SimpleITK as sitk
from src import config
from src.tools import resample_image

def canny_filtering(input_image, 
                    output_image, 
                    resample_size=None,
                    mri=False,
                    variance=2,
                    thresh=None             
                    ):
    '''
    Apply Canny contour detection algorithm. Resample image if necessary.

    Args:
        input_image (str): input image path
        output_image (str): output image path
        resample_size (int): if used, input volume is resampled along x and y with this size 
        mri (bool): if true, volume intensity is normalized
        variance (int): variance used in Canny Filter
        thresh (dict): threshold values used in Canny Filter
    '''

    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(input_image)
    image = reader.Execute()

    t0 = time.time()

    if image.GetPixelIDTypeAsString() !=  '32-bit float':
        castImageFilter = sitk.CastImageFilter()
        castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
        image = castImageFilter.Execute(image)

    if mri:
        normalizer = sitk.NormalizeImageFilter()
        image = normalizer.Execute(image)

    if resample_size:
        image = resample_image(image, resample_size)
    
    cannyFilter = sitk.CannyEdgeDetectionImageFilter()
    cannyFilter.SetVariance(variance)
    cannyFilter.SetLowerThreshold(thresh['low'])
    cannyFilter.SetUpperThreshold(thresh['up'])
    cannyImg = cannyFilter.Execute(image)

    rescaler = sitk.RescaleIntensityImageFilter()
    rescaler.SetOutputMaximum(255)
    rescaler.SetOutputMinimum(0)
    rescaledImg= rescaler.Execute(cannyImg)

    t1 = time.time()

    writer = sitk.ImageFileWriter()
    writer.SetFileName(output_image)
    writer.Execute(rescaledImg)

    print('Canny filtering applied with the thresholds', thresh['low'], thresh['up'], 'and variance ', variance)
    array = sitk.GetArrayFromImage(rescaledImg)
    xyz = np.argwhere(array>0)
    print( str( len( xyz ) )  + " points detected" )
    print('Computation time:' + str(t1-t0))


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, help='Configuration file with details to run code', required=True)
    parser.add_argument('--id', help='Process only this id')
    parser.add_argument('-s', "--skip_existing", help='Skip computation when file already exists', action = "store_true")
    args = parser.parse_args()

    root_path = os.path.dirname(os.path.realpath(__file__))
    cfg = config.load_config(args.config, os.path.join(root_path, 'configs/default.yaml'))

    dataset_path = os.path.join(cfg['data']['data_path'], cfg['data']['dataset'])
    canny_path = os.path.join(dataset_path,'canny',cfg['data']['canny_path'])
    img_dir = os.path.join(dataset_path,'volumes')
    if not os.path.exists(canny_path):
        os.makedirs(canny_path)

    thresh = cfg['data']['canny_threshold']

    csv_path = os.path.join(dataset_path,'info_dataset.csv')
    df = pd.read_csv(csv_path)
    df = df.reset_index(drop=True)
    
    for index, row in df.iterrows():
        id = row['id']
        if args.id and args.id != id : continue
        print(id)
        dim_z = row['dim_z']

        input_path = os.path.join(img_dir, id + '.nii.gz')
        output_path = os.path.join(canny_path, id + '.nii.gz')

        if args.skip_existing and os.path.exists( output_path ):
            print( "Canny volume already exists, skipping" )
            continue

        resample_size=None
        if cfg['data']['canny_resolution'] > 0:
            resolution = cfg['data']['canny_resolution']
            resample_size =(resolution,resolution,dim_z)
        print( "Resample : " + str( resample_size ) )
        canny_filtering(input_path,
                        output_path,
                        resample_size=resample_size,
                        mri=cfg['data']['mri'],
                        variance=cfg['data']['canny_variance'],
                        thresh=thresh,
                        )
