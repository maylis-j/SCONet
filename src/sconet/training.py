import torch
from src.common import get_inputs
from src.losses import CELoss   
from src.metrics import (compute_dice, compute_iou)
from src.training import BaseTrainer

import math

class Trainer(BaseTrainer):
    ''' Trainer object for the Occupancy Network.

    Args:
        model (nn.Module): Occupancy Network model
        optimizer (optimizer): pytorch optimizer object
        device (device): pytorch device
        threshold (float): threshold value
    '''

    def __init__(self, model, optimizer, device=None,
                 threshold=0.5, metric='dice', loss=CELoss,
                 compute_metric_on_training_set=True,
                ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.threshold = threshold
        self.metric = metric
        self.loss = loss
        self.compute_metric_on_training_set = compute_metric_on_training_set

    def train_step(self, data):
        ''' Performs a training step.

        Args:
            data (dict): data dictionary
        '''
        self.model.train()
        self.optimizer.zero_grad()
        eval_dict = self.compute_loss(data)
        eval_dict['loss'].backward()
        self.optimizer.step()
        
        eval_dict['loss'] = eval_dict['loss'].item()
        if self.compute_metric_on_training_set:
            eval_dict[self.metric] = self.compute_metric(data)
        return eval_dict
    

    def eval_step(self, data):
        ''' Performs an evaluation step.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold
        eval_dict = {}

        points = data.get('points').to(device)
        occ = data.get('points.occ').to(device)

        points_metric = data.get('points_metric').to(device)
        occ_metric = data.get('points_metric.occ').to(device)

        inputs = get_inputs(data)
        inputs = inputs.to(device)
        kwargs = {}
        
        # Compute loss
        with torch.no_grad():
            c = self.model.encode_inputs(inputs)
            logits = self.model.decode(points, c, **kwargs)
        eval_dict['loss'] = self.loss(logits,occ).mean()


        # Compute metric
        with torch.no_grad():
            p_out = self.model(points_metric, inputs, **kwargs)
        p_out = torch.nn.functional.softmax(p_out,dim=2)
        occ_metric_np = (occ_metric >= 0.5).cpu().numpy()
        occ_metric_hat_np = (p_out >= threshold).cpu().numpy()
        if self.metric=='iou':
            metric = compute_iou(occ_metric_np, occ_metric_hat_np).mean(axis=0)
        elif self.metric=='dice':
            metric = compute_dice(occ_metric_np, occ_metric_hat_np)
        eval_dict[self.metric] = metric
        
        return eval_dict



    def compute_loss(self, data):
        ''' Computes the loss and metric during training.

        Args:
            data (dict): data dictionary
        '''
        device = self.device
        eval_dict = {}

        p = data.get('points').to(device)
        occ = data.get('points.occ').to(device)
        
        inputs = get_inputs(data)
        inputs = inputs.to(device)
        kwargs = {}
                
        # Compute loss
        c = self.model.encode_inputs(inputs)
        logits = self.model.decode(p, c, **kwargs)

        eval_dict['loss'] = self.loss(logits,occ).mean()
        
        return eval_dict
    
    def compute_metric(self, data):
        ''' Computes the loss and metric during training.

        Args:
            data (dict): data dictionary
        '''
        self.model.eval()

        device = self.device
        threshold = self.threshold

        points_metric = data.get('points_metric').to(device)
        occ_metric = data.get('points_metric.occ').to(device)
        
        inputs = get_inputs(data)
        inputs = inputs.to(device)
        kwargs = {}

        # Compute metric
        with torch.no_grad():
            p_out = self.model(points_metric, inputs, **kwargs)
        p_out = torch.nn.functional.softmax(p_out,dim=2)
        occ_metric_np = (occ_metric >= 0.5).cpu().numpy()
        occ_metric_hat_np = (p_out >= threshold).cpu().numpy()
        if self.metric=='iou':
            metric = compute_iou(occ_metric_np, occ_metric_hat_np).mean(axis=0)
        elif self.metric=='dice':
            metric = compute_dice(occ_metric_np, occ_metric_hat_np)

        return metric
