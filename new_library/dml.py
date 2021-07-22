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
        elif loss_non_linearity_name.lower() == 'tanh':
            self.loss_non_linearity = nn.TanH()
        elif loss_non_linearity_name.lower() == 'htanh':
            self.loss_non_linearity = nn.Hardtanh()
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
        labels,
    ):
        degradation_metric_1 = self.forward(x1)
        degradation_metric_2 = self.forward(x2)
        
        predicted_labels = self.compute_predicted_label(degradation_metric_1, degradation_metric_2)
                                      
        loss = self.criterion(predicted_labels, labels)                 
        return loss

class RefBasedDeepMetric(nn.Module):
    
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
    
    def compute_binary_loss(self,
        predicted_labels,
        labels
    ):
        return self.criterion(predicted_labels, labels)
    
    def compute_loss(self,
        x1,
        x2,
        refs,
        labels,
    ):
        
        degradation_metric_1 = self.forward(x1, refs)
        degradation_metric_2 = self.forward(x2, refs)
        
        predicted_labels = self.compute_predicted_label(degradation_metric_1, degradation_metric_2)
        
        loss = self.criterion(predicted_labels, labels)
        preds = torch.gt(predicted_labels,0.5).float()
        
        return loss, preds
        
        

        
        