import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils import data

from src.tools import get_affine

logger = logging.getLogger(__name__)

# Fields
class Field(object):
    ''' Data fields class.
    '''

    def load(self, info):
        ''' Loads a data point.

        Args:
            info (dict): dict with config information on the subject
        '''
        raise NotImplementedError



class Shapes3dDataset(data.Dataset):
    ''' 3D Shapes dataset class.
    '''

    def __init__(self, fields, split=None,
                 no_except=True, transform=None, cfg=None):
        ''' Initialization of the the 3D shape dataset.

        Args:
            fields (dict): dictionary of fields
            split (str): which split is used
            no_except (bool): no exception
            transform (callable): transformation applied to data points
            cfg (yaml): config file
        '''
        # Attributes
        self.fields = fields
        self.no_except = no_except
        self.transform = transform
        self.cfg = cfg
        self.dataset_folder = os.path.join(cfg['data']['data_path'],cfg['data']['dataset'])
        
        df_info = pd.read_csv(os.path.join(self.dataset_folder, 'info_dataset.csv'))
        
        # Get all subjects
        self.subjects = []

        if split is None:
            self.subjects += [
                {'subject': m} for m in [d for d in os.listdir(self.dataset_folder) \
                                         if (os.path.isdir(os.path.join(self.dataset_folder, d)) and d != '') ]
            ]

        else:
            split_file = os.path.join(self.dataset_folder, split + '.lst')
            with open(split_file, 'r') as f:
                subjects_c = f.read().split('\n')
                
            if '' in subjects_c:
                subjects_c.remove('')
            
            for m in subjects_c:
                subjectname = m.split(',')[0]
                row = df_info[df_info['id']==subjectname]
                affine = get_affine(row)
                shape = [row['dim_x'].values[0], row['dim_y'].values[0], row['dim_z'].values[0]]
                spacing = [row['spacing_x'].values[0], row['spacing_y'].values[0], row['spacing_z'].values[0]]
                dim = [row['dim_x'].values[0], row['dim_y'].values[0], row['dim_z'].values[0]]
                info_dict = {'subject': subjectname, 
                             'affine': affine, 
                             'shape': shape, 
                             'spacing': spacing,
                             'dim': dim}
                self.subjects += [
                    info_dict    
                ]

            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return len(self.subjects)

    def __getitem__(self, idx):
        ''' Returns an item of the dataset.

        Args:
            idx (int): ID of data point
        '''
        subject = self.subjects[idx]['subject']
        affine = self.subjects[idx]['affine']
        shape = self.subjects[idx]['shape']
        spacing = self.subjects[idx]['spacing']
        dim = self.subjects[idx]['dim']

        subject_path = os.path.join(self.dataset_folder, 'pointcloud', subject)
        info = {
            'subject_path':subject_path,
            'idx':idx,
            'affine':affine,
            'shape':shape,
            'spacing': spacing,
            'dim': dim
            }

        data = {}
        for field_name, field in self.fields.items():
            logging.debug(field)
            try:
                field_data = field.load(info)
            except Exception as e:
                print(subject)
                print(str(e))
                if self.no_except:
                    logger.warn(
                        'Error occured when loading field %s of subject %s'
                        % (field_name, subject)
                    )
                    return None
                else:
                    raise

            if isinstance(field_data, dict):
                for k, v in field_data.items():
                    if k is None:
                        data[field_name] = v
                    else:
                        data['%s.%s' % (field_name, k)] = v
            else:
                data[field_name] = field_data
        if self.transform is not None:
            data = self.transform(data)

        return data

    
    def get_subject_dict(self, idx):
        return self.subjects[idx]
    

def collate_remove_none(batch):
    ''' Collater that puts each data field into a tensor with outer dimension
        batch size.

    Args:
        batch: batch
    '''

    batch = list(filter(lambda x: x is not None, batch))
    return data.dataloader.default_collate(batch)


def worker_init_fn(worker_id):
    ''' Worker init function to ensure true randomness.
    '''
    def set_num_threads(nt):
        try: 
            import mkl; mkl.set_num_threads(nt)
        except: 
            pass
            torch.set_num_threads(1)
            os.environ['IPC_ENABLE']='1'
            for o in ['OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:
                os.environ[o] = str(nt)

    random_data = os.urandom(4)
    base_seed = int.from_bytes(random_data, byteorder="big")
    np.random.seed(base_seed + worker_id)
