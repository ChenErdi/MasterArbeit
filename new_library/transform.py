import numpy as np
from PIL import Image


class RandomFlip(object):
    
    def __init__(self, lr_prob=0.5, ud_prob=0.5, channels=None):
        if lr_prob > 1 or lr_prob < 0 or ud_prob > 1 or ud_prob < 0:
            raise ValueError('Invalid flip probability')
            
        self.lr_prob = lr_prob
        self.ud_prob = ud_prob
                
        #if channels is not None:
        #    self.channels = channels
        
    def __call__(self, img):
        
        for i_channel in range(img.shape[2]):
                
            if np.random.rand() <= self.lr_prob:
                img[:, :, i_channel] = np.fliplr(img[:, :, i_channel])
                
            if np.random.rand() <= self.ud_prob:
                img[:, :, i_channel] = np.flipud(img[:, :, i_channel])
                
        return img
                
class RandomOffset(object):
    
    def __init__(self, min_offset=-5, max_offset=5, channels=None):
                
        self.min_offset = min_offset
        self.max_offset = max_offset
                
        if channels is not None:
            self.channels = channels
        
    def __call__(self, img):
        
        for i_channel in range(img.shape[2]):
            min_val = np.maximum(
                self.min_offset, 
                -np.amin(img[:, :, i_channel])
            )
                             
            img[:, :, i_channel] = img[:, :, i_channel] + np.random.randint(min_val, self.max_offset)
                
        return img
    
class RandomRotateGrayscale(object):
    
    def __init__(self, fill=140, rot_range=[-0.1, 0.1]):
                
        self.rot_range = rot_range
        self.fill = fill
        
        self._offs = np.min(rot_range)
        self._range = np.max(rot_range) - self._offs
        
    def __call__(self, img):
        
        exp_dim = False
        if img.ndim == 3:
            img = img[:, :, 0]
            exp_dim = True
            
        img = Image.fromarray(img)
        val = self._offs + np.random.rand() * self._range
        
        img = img.rotate(val, resample=Image.BILINEAR, fillcolor=self.fill)
        
        img = np.array(img)
        
        if exp_dim:
            img = np.expand_dims(img, 2)
                
        return img
        