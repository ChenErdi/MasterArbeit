import os
import json

import numpy as np
import scipy
import sklearn
import pandas as pd
import cv2
from PIL import Image

import torch
from torch import nn
import torchvision
from torchvision import models, transforms
from .cresnet import initialize_cresnet

class RefFreeDeepMetric(nn.Module):
    
    def __init__(self,
        model,
        loss_non_linearity_name='htanh',
        criterion_name='bce',
    ):
        
        super(RefFreeDeepMetric, self).__init__()
        
        self.model = model
        
        # val_range_limiter can be sigmoid or tanh depending on labels
        if loss_non_linearity_name.lower() == 'sigmoid':
            self.loss_non_linearity = nn.Sigmoid()
        elif loss_non_linearity_name.lower() == 'silu' or loss_non_linearity_name.lower() == 'swish':
            self.loss_non_linearity = nn.SiLU()
        elif loss_non_linearity_name.lower() == 'hardswish':
            self.loss_non_linearity = nn.Hardswish()
        elif loss_non_linearity_name.lower() == 'tanh':
            self.loss_non_linearity = nn.TanH()
        elif loss_non_linearity_name.lower() == 'htanh':
            self.loss_non_linearity = nn.Hardtanh()
        elif loss_non_linearity_name.lower() == 'htanh01':
            self.loss_non_linearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        else:
            raise NotImplementedError("{} is not implemented".format(loss_non_linearity_name))
            
        if criterion_name.lower() == 'bce':
            # By default the criterion is BCE loss for binary class
            self.criterion = nn.BCELoss()
        elif criterion_name.lower() == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError("{} is not implemented".format(criterion_name))
        
    def compute_predicted_label(self,
        degradation_metric_1,
        degradation_metric_2,
    ):
        diff = degradation_metric_1 - degradation_metric_2
        
        predicted_labels = self.loss_non_linearity(diff)
        
        return predicted_labels        
        
    def forward(self,
        x
    ):
        return self.model(x)
    
    def compute_loss(self,
        x1,
        x2,
        y1,
        y2,
    ):
        labels = self.compute_predicted_label(y1, y2)
        
        degradation_metric_1 = self.forward(x1)
        degradation_metric_2 = self.forward(x2)
        
        predicted_labels = self.compute_predicted_label(degradation_metric_1, degradation_metric_2)
                                      
        loss = self.criterion(predicted_labels, labels)                 
        return loss

class RefBasedDeepMetric(nn.Module):
    
    @classmethod
    def load_pretrained_cresnet(cls, model_path, device, show_config=False):
        with open(os.path.join(model_path, 'config.json')) as f:
            config = json.load(f)
            
        if show_config:
            print(config)
    
        weight_fpath = os.path.join(model_path, 'model.weight')
        loss_non_linearity_name=config['loss_non_linearity_name']
        criterion_name=config['criterion_name']
        
        cresnet_model = initialize_cresnet(config['cresnet_config'], device)
        
        model = cls(
            cresnet_model, 
            loss_non_linearity_name=loss_non_linearity_name,
            criterion_name=criterion_name,
        )
        model.load_state_dict(torch.load(weight_fpath))
        
        model.eval()

        return model
    
    def __init__(self,
        feature_extractor,
        loss_non_linearity_name='sigmoid',
        criterion_name='bce',
    ):
        super(RefBasedDeepMetric, self).__init__()
                                      
        self.feature_extractor = feature_extractor
        
        # val_range_limiter can be sigmoid or tanh depending on labels
        if loss_non_linearity_name.lower() == 'sigmoid':
            self.loss_non_linearity = nn.Sigmoid()
        elif loss_non_linearity_name.lower() == 'tanh':
            self.loss_non_linearity = nn.TanH()
        elif loss_non_linearity_name.lower() == 'htanh':
            self.loss_non_linearity = nn.Hardtanh()
        elif loss_non_linearity_name.lower() == 'htanh01':
            self.loss_non_linearity = nn.Hardtanh(min_val=0.0, max_val=1.0)
        else:
            raise NotImplementedError("{} is not implemented".format(loss_non_linearity_name))
            
        if criterion_name.lower() == 'bce':
            # By default the criterion is BCE loss for binary class
            self.criterion = nn.BCELoss()
        elif criterion_name.lower() == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise NotImplementedError("{} is not implemented".format(criterion_name))
                                      
    def compute_degradation_metric(self,
        vin:torch.Tensor,
        vrefs:torch.Tensor,
    ):
        num_ref = vrefs.size(0)
        batch_size = vin.size(0)
        
        repeated_vin = vin.unsqueeze(1).repeat(1, num_ref, 1)
        repeated_vrefs = vrefs.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Compute squared difference
        # Output shape: [batch_size, num_ref]
        kernel_mat = (repeated_vin-repeated_vrefs).pow(2).sum(2) #torch.Size([batchsize, 10])

        # Compute degradation metric (Mean Squard Difference/Error between vin and vrefs)
        # Output shape: [batch_size]
        degradation_metric = torch.mean(kernel_mat,dim=1) #torch.Size([batchsize])
        
        return degradation_metric
    
    def compute_predicted_label(self,
        degradation_metric_1,
        degradation_metric_2,
    ):
        """
        
        """
        
        diff = degradation_metric_1 - degradation_metric_2
        
        predicted_labels = self.loss_non_linearity(diff)
        
        return predicted_labels
    
    def forward(self,
        x:torch.Tensor,
        refs:torch.Tensor,
    ):

        # Compute latent vectors v*
        vin = self.feature_extractor(x)
        vrefs = self.feature_extractor(refs)
        
        return self.compute_degradation_metric(vin, vrefs)
    
    def featuremap(self,x:torch.Tensor,):
        # Compute latent vectors v*
        
        vin = self.feature_extractor(x)
        
        return self.feature_extractor.featuremapfc
    
    def featuremap_refs(self,refs:torch.Tensor,):
        # Compute refs latent vectors v*
        vrefs = self.feature_extractor(refs)
        
        return self.feature_extractor.featuremapfc
    
    def compute_binary_loss(self,
        predicted_labels,
        labels
    ):
        return self.criterion(predicted_labels, labels)
    
    def compute_loss(self,
        x1:torch.Tensor,
        x2:torch.Tensor,
        refs:torch.Tensor,
        targets:torch.Tensor,
    ):
        
        degradation_metric_1 = self.forward(x1, refs)
        degradation_metric_2 = self.forward(x2, refs)

        predicted_labels = self.compute_predicted_label(degradation_metric_1, degradation_metric_2)
        
        loss = self.criterion(predicted_labels, targets)
        preds = torch.gt(predicted_labels,0.5).float()
        
        return loss, preds
        
        

        
        