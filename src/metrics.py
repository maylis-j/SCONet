import numpy as np
import SimpleITK as sitk
from src.tools import keep_one_label

def compute_iou(occ1, occ2):
    ''' Computes the Intersection over Union (IoU) value for two sets of
    occupancy values.

    Args:
        occ1 (tensor): first set of occupancy values
        occ2 (tensor): second set of occupancy values
    '''
    occ1 = np.asarray(occ1)
    occ2 = np.asarray(occ2)

    # Convert to boolean values
    occ1 = (occ1 >= 0.5)
    occ2 = (occ2 >= 0.5)

    # Compute IOU
    area_union = (occ1 | occ2).astype(np.float32).sum(axis=1)
    area_intersect = (occ1 & occ2).astype(np.float32).sum(axis=1)
    iou = (area_intersect / area_union)


    return iou

def compute_dice(occ1,occ2):
    '''
    Computes Dice metric over two volumes for all organs.
    '''
    epsilon = 1e-5
    num_classes = occ1.shape[2]
    dice = []
    for l in range(num_classes):
        targets_l = occ1[:,:,l]
        inputs_l = occ2[:,:,l]
        intersection = (inputs_l * targets_l).sum()
        dice += [(2.*intersection + epsilon)/(inputs_l.sum() + targets_l.sum() + epsilon)]
    return dice


def compute_hausdorff(image1, image2, classes):
    '''
    Computes Hausdorff distance (HD) between two volumes.
    Args:
        image1 (object): target sitk image
        image2 (object): predicted sitk image
        classes (list): names of all the classes to process
    
    Returns:
        hd (dict): dictionnary with HD values for each organ, and the 
        mean HD value.

    '''
    num_classes = len(classes)
    hd = {'hd_tot':0}

    hd_tot = 0
    count = 0

    for l in range(1,num_classes):
        
        image1_l = keep_one_label(image1,l,num_classes)
        image2_l = keep_one_label(image2,l,num_classes)

        try:
            haussdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
            haussdorff_distance_filter.Execute(image1_l,image2_l)
            hd_val = haussdorff_distance_filter.GetHausdorffDistance()

            hd['hd_'+classes[l]] = hd_val
            hd_tot += hd_val
            count += 1
        except Exception as e:
            print(classes[l],"not found -> Hausdorff failed.")
    
    if count == 0:
        print("All Hausdorff failed...")
        hd['hd_tot'] = 1000
    else:
        hd['hd_tot'] = hd_tot/count
    
    return hd


def compute_dice_per_organ(target, inputs, classes):
    '''
    Computes Dice metric over two volumes.
    Args:
        target (array): target numpy array
        inputs (array): predicted numpy array
        classes (list): names of all the classes to process
    
    Returns:
        dice (dict): dictionnary with Dice values for each organ, and the 
        mean Dice value. The background label is not included.

    '''
    num_classes = len(classes)
    dice={}
    epsilon = 1e-5
    dice_counter = 0
    for l in range(1, num_classes):
        inputs_l = (inputs==l)*1
        targets_l = (target==l)*1
        intersection = (inputs_l * targets_l).sum()
        dice_l = (2.*intersection + epsilon)/(inputs_l.sum() + targets_l.sum() + epsilon)
        dice['dice_'+classes[l]] = dice_l
        dice_counter += dice_l
    dice['dice_tot'] = dice_counter/(num_classes-1)
    return dice
