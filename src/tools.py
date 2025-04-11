import nibabel as nib
import numpy as np
import SimpleITK as sitk

def read_sitk(input_path):
    '''
    Reads Nifti image using SimpleITK library.
    
    Args:
        input_path (str): path to input image
    
    Returns:
        object: sitk image
    '''
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(input_path)
    image = reader.Execute()
    return image

def load_sitk(input_path, dim = None):
    '''
    Loads sitk image as a numpy array and swap axes to have correct x, y and z.

    Args:
        input_path (str): path to input image
        dim (list): if given, the image is resampled to the given dimension

    Returns:
        Sitk image and its corresponding numpy array

    '''
    img = read_sitk(input_path)

    if ( dim is not None ) and ( dim != img.GetSize() ):
        img = resample_image(img, dim)
    array = sitk.GetArrayFromImage(img)
    array = np.swapaxes(array,0,2)

    return img, array

def resample_image(image, out_size, out_spacing=None):
    '''
    Resample the image to fit given size.

    Args:
        image (object): Nifti image
        out_size (list): desired output size along x,y and z
        out_spacing (list): if given, desired output spacing along x,y and z
    
    Returns:
        object: the resampled Nifti image
    '''
    
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
    resample_filter.SetOutputDirection(image.GetDirection())
    resample_filter.SetOutputOrigin(image.GetOrigin())
    
    input_size = image.GetSize()
    spacing = image.GetSpacing()
    if out_spacing is None:
        out_spacing = [i *j / k for i, j, k in zip(spacing, input_size, out_size)]

    resample_filter.SetOutputSpacing(out_spacing)
    resample_filter.SetSize(np.array(out_size, dtype='int').tolist()) # doesn't work without this weird workaround

    new_img = resample_filter.Execute(image)

    return new_img

def get_affine(row, dim = None):
    '''
    Gets the affine matrix from a row of info_dataset.csv

    Args:
        row (dict): row read from info_dataset.csv
        dim (list): if given, the affine matrix is modified to match the resampled image

    Returns:
        Affine matrix
    '''
    affine = [row['affine_'+str(i)] for i in range(16)]
    affine = np.reshape(affine, (4,4))
    if dim is not None:
        affine[0][0] = affine[0][0] * row['dim_x'] / dim[0]
        affine[1][1] = affine[1][1] * row['dim_y'] / dim[1]
        affine[2][2] = affine[2][2] * row['dim_z'] / dim[2]
    return affine

def to_nii(array, out_path, affine):
    if len(array.shape)==4:
        data_seg = np.argmax(array, axis=3).astype(np.int16)
    else:
        data_seg = array.astype(np.int16)
    img = nib.Nifti1Image(data_seg,affine)
    nib.save(img, out_path)

def keep_one_label(image, l, num_labels):
    '''
    Sets all labels except one to be background labels. 
    Args:
        image (object): sitk image to process
        l (int): label to keep
        num_labels (int): original number of labels
    '''
    other_labels = [j for j in range(1,num_labels) if j!= l]
    mapping = {}
    for j in other_labels:
        mapping[j] = 0

    image_l = change_label_nii(image, mapping)
    
    return image_l

def change_label_nii(img_nii,mapping):
    change_label_filter = sitk.ChangeLabelImageFilter()
    change_label_filter.SetChangeMap(mapping)
    img_new = change_label_filter.Execute(img_nii)
    return img_new