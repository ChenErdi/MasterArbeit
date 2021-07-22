from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import numpy as np
import os
import h5py
import logging as log

'''
DatasetH5Two Random are used to create dataset by randomly select two images. The target is 0 or 1.  

'''
class DatasetH5TwoRandom(Dataset):
    
    def __init__(self,
        h5_fpath,
        num_ref = 10,
        select_num_img = None,      
        transform = None,
        label_transform = None
        
    ):

        self.h5_fpath = h5_fpath
        self.num_ref = num_ref
        self.transform = transform
        self.label_transform = label_transform
        

        self._h5_f = h5py.File(self.h5_fpath, mode='r') # the h5 file shape is (2001,720,20)

        self.label = self._h5_f['label'] # label shape: (2001,)

        self.mean = self._h5_f['mean'] # <HDF5 dataset "mean": shape (720, 20)

        self.raw_imgs = self._h5_f['img'] # <HDF5 dataset "img": shape (2001,720, 20)

        self.imgs = np.expand_dims(self.raw_imgs,-1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,720, 20, 1) 
        # self.img = np.expand_dims(self.raw_img,1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,1, 720, 20) 

        self.img_total_num = len(self.label)

        self.all_idx = np.arange(self.img_total_num) # 0-2000 1-D ndarray

    # get first n images as the references.
    def getRef(self):
        refs = self.imgs[:self.num_ref]  
        refs_list = []

        if self.transform is not None:
            for im in refs:
                ref = self.transform(im)
                ref = ref.float()
                refs_list.append(ref)
                
        refs_tensor = torch.stack(refs_list,dim = 0)
        return refs_tensor

    # Return the length of selected num of images
    def __len__(self):
        return self.img_total_num - self.num_ref

    # Get label and data of images 
    def __getitem__(self,idx):
        """
        Args:
            index (int): Index
        Returns:
            Random img1, img2, and if label1 < label2 then target = 0
        """
        # avoid to select the references
        idx_avail = self.all_idx[self.num_ref:]

        # randomly get img1 and img2
        label1 = int(np.random.choice(idx_avail,1)) # avoid to get reference image
        while True:
            label2 = int(np.random.choice(idx_avail,1))
            if label1 != label2:
                break
                
        img1 = self.imgs[label1,:,:]
        img2 = self.imgs[label2,:,:]

        # if image1 is older than image2, then label = 1
        if label1 > label2: 
            label = 1.0
        else:
            label = 0.0
        label = torch.from_numpy(np.array([label], dtype=np.float32))

        #img = self.img[idx + self.num_ref]
        #label = self._h5_f["label"][idx + self.num_ref].astype(np.float)

        if self.transform is not None:
            img1 = self.transform(img1)
            img1 = img1.float()

            img2 = self.transform(img2)
            img2 = img2.float()

        if self.label_transform is not None:
            label = self.label_transform(label)
            label = label.float()

        return img1, img2, label
    
    def __del__(self):
        self._h5_f.close()

r'''
DatasetH5ForTest used to prepare dataset for ploting degradation metrics, it return one image and label.

'''
class DatasetH5ForTest(Dataset):
    
    def __init__(self,
        h5_fpath,
        num_ref = 10,
        select_num_img = None,      
        transform = None,
        label_transform = None,
        normalize_im=True,
        single_mean_val=True,
        
    ):

        self.h5_fpath = h5_fpath
        self.num_ref = num_ref
        self.transform = transform
        self.label_transform  = label_transform
        self.normalize_im = normalize_im
        self.single_mean_val = single_mean_val
        

        self._h5_f = h5py.File(self.h5_fpath, mode='r') # the h5 file shape is (2001,720,20)

        self.label = self._h5_f['label'] # label shape: (2001,)

        self.mean = self._h5_f['mean'] # <HDF5 dataset "mean": shape (720, 20)

        self.raw_imgs = self._h5_f['img'] # <HDF5 dataset "img": shape (2001,720, 20)

        self.imgs = np.expand_dims(self.raw_imgs,-1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,720, 20, 1) 
        # self.img = np.expand_dims(self.raw_img,1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,1, 720, 20) 

        self.img_total_num = len(self.label)

        self.all_idx = np.arange(self.img_total_num) # 0-2000 1-D ndarray

    # get first n images as the references.
    def getRef(self):
        refs = self.imgs[:self.num_ref]  
        refs_list = []

        if self.transform is not None:
            for im in refs:
                ref = self.transform(im)
                ref = ref.float()
                refs_list.append(ref)
                
        refs_tensor = torch.stack(refs_list,dim = 0)
        return refs_tensor

    # Return the length of selected num of images
    def __len__(self):
        return self.img_total_num - self.num_ref

    # Get label and data of images 
    def __getitem__(self,idx):
        """
        Args:
            index (int): Index
        Returns:
            img, label
        """
        # avoid to select the references
        idx_avail = self.all_idx[self.num_ref:]
        idx = idx_avail[idx]

        im = self.imgs[idx,:,:]

        if self.normalize_im:
            if self.single_mean_val:
                im = im.astype(np.float32) - self._h5_f["mean"][idx, :, :].mean()
            else:
                im = im.astype(np.float32) - self._h5_f["mean"][idx, :, :]
        
        #im = np.expand_dims(im, 1)
        #im = im.astype(np.float32)
        
        if self.transform is not None:
            im = self.transform(im)
            im = im.float()
            
        label = self.label[idx].astype(np.float)
        if self.label_transform is not None:
            label = self.label_transform(label)
            label = label.float()
            
        return im, label

    def __del__(self):
        self._h5_f.close()
        
r'''
DatasetH5TwoRandomV2 Drop the first 200 images of each sequence to ensure the quality of the reference image

'''
        
class DatasetH5TwoRandomV2(Dataset):
    
    def __init__(self,
        h5_fpath,
        num_ref = 10,
        select_num_img = None,      
        transform = None,
        label_transform = None
        
    ):

        self.h5_fpath = h5_fpath
        self.num_ref = num_ref
        self.transform = transform
        self.label_transform = label_transform
        

        self._h5_f = h5py.File(self.h5_fpath, mode='r') # the h5 file shape is (2001,720,20)

        self.label = self._h5_f['label'][200:] # Drop the first 500 images

        self.mean = self._h5_f['mean'][200:] # <HDF5 dataset "mean": shape (720, 20)

        self.raw_imgs = self._h5_f['img'][200:] # Drop the first 200 images of each sequence to ensure the quality of the reference image

        self.imgs = np.expand_dims(self.raw_imgs,-1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,720, 20, 1) 
        # self.img = np.expand_dims(self.raw_img,1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,1, 720, 20) 

        self.img_total_num = len(self.label)

        self.all_idx = np.arange(self.img_total_num) # 0-2000 1-D ndarray

    # get first n images as the references.
    def getRef(self):
        refs = self.imgs[:self.num_ref]  
        refs_list = []

        if self.transform is not None:
            for im in refs:
                ref = self.transform(im)
                ref = ref.float()
                refs_list.append(ref)
                
        refs_tensor = torch.stack(refs_list,dim = 0)
        return refs_tensor

    # Return the length of selected num of images
    def __len__(self):
        return self.img_total_num - self.num_ref

    # Get label and data of images 
    def __getitem__(self,idx):
        """
        Args:
            index (int): Index
        Returns:
            Random img1, img2, and if label1 < label2 then target = 0
        """
        # avoid to select the references
        idx_avail = self.all_idx[self.num_ref:]

        # randomly get img1 and img2
        label1 = int(np.random.choice(idx_avail,1)) # avoid to get reference image
        while True:
            label2 = int(np.random.choice(idx_avail,1))
            if label1 != label2:
                break
                
        img1 = self.imgs[label1,:,:]
        img2 = self.imgs[label2,:,:]

        # if image1 is older than image2, then label = 1
        if label1 > label2: 
            label = 1.0
        else:
            label = 0.0
        label = torch.from_numpy(np.array([label], dtype=np.float32))

        #img = self.img[idx + self.num_ref]
        #label = self._h5_f["label"][idx + self.num_ref].astype(np.float)

        if self.transform is not None:
            img1 = self.transform(img1)
            img1 = img1.float()

            img2 = self.transform(img2)
            img2 = img2.float()

        if self.label_transform is not None:
            label = self.label_transform(label)
            label = label.float()

        return img1, img2, label
    
    def __del__(self):
        self._h5_f.close()

r'''
DatasetH5ForTestV2 Drop the first 200 images of each sequence to ensure the quality of the reference image

'''
class DatasetH5ForTestV2(Dataset):
    
    def __init__(self,
        h5_fpath,
        num_ref = 10,
        select_num_img = None,      
        transform = None,
        label_transform = None,
        normalize_im=True,
        single_mean_val=True,
        
    ):

        self.h5_fpath = h5_fpath
        self.num_ref = num_ref
        self.transform = transform
        self.label_transform  = label_transform
        self.normalize_im = normalize_im
        self.single_mean_val = single_mean_val
        

        self._h5_f = h5py.File(self.h5_fpath, mode='r') # the h5 file shape is (2001,720,20)

        self.label = self._h5_f['label'][200:] # label shape: (2001,)

        self.mean = self._h5_f['mean'][200:] # <HDF5 dataset "mean": shape (720, 20)

        self.raw_imgs = self._h5_f['img'][200:] # Drop the first 200 images of each sequence to ensure the quality of the reference image

        self.imgs = np.expand_dims(self.raw_imgs,-1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,720, 20, 1) 
        # self.img = np.expand_dims(self.raw_img,1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,1, 720, 20) 

        self.img_total_num = len(self.label)

        self.all_idx = np.arange(self.img_total_num) # 0-2000 1-D ndarray

    # get first n images as the references.
    def getRef(self):
        refs = self.imgs[:self.num_ref]  
        refs_list = []

        if self.transform is not None:
            for im in refs:
                ref = self.transform(im)
                ref = ref.float()
                refs_list.append(ref)
                
        refs_tensor = torch.stack(refs_list,dim = 0)
        return refs_tensor

    # Return the length of selected num of images
    def __len__(self):
        return self.img_total_num - self.num_ref

    # Get label and data of images 
    def __getitem__(self,idx):
        """
        Args:
            index (int): Index
        Returns:
            img, label
        """
        # avoid to select the references
        idx_avail = self.all_idx[self.num_ref:]
        idx = idx_avail[idx]

        im = self.imgs[idx,:,:]

        if self.normalize_im:
            if self.single_mean_val:
                im = im.astype(np.float32) - self._h5_f["mean"][idx, :, :].mean()
            else:
                im = im.astype(np.float32) - self._h5_f["mean"][idx, :, :]
        
        #im = np.expand_dims(im, 1)
        #im = im.astype(np.float32)
        
        if self.transform is not None:
            im = self.transform(im)
            im = im.float()
            
        label = self.label[idx].astype(np.float)
        if self.label_transform is not None:
            label = self.label_transform(label)
            label = label.float()
            
        return im, label

    def __del__(self):
        self._h5_f.close()

def compute_std_mean(seq_idx_list, path):
    
    """
    Args:
        seq_idx_list (list of int): To get the sequence of the specified number
        path(String): the path of HDF5 Folder
    Returns:
        mean_val , std_val
    """
    mean_list = []
    std_list = []

    for seq_idx in seq_idx_list:
        h5_fpath = path.format(seq_idx)
        f = h5py.File(h5_fpath,"r")
        images = f['img']

        data = [d for d in images]
        mean = np.mean(data)
        mean_list.append(mean)
        std = np.std(data)
        std_list.append(std)

    mean_val = sum(mean_list) / len(mean_list)
    std_val = sum(std_list) / len(std_list)
    
    return mean_val , std_val

class TROIDatasetH5(Dataset):
    def __init__(self,
        h5_fpath,
        normalize_im=True,
        single_mean_val=True,
        roi_idx=None,
        random_translation=True,
        im_transforms = None,
        label_transforms = None,
        mask=None
    ):
        
        self.h5_fpath = h5_fpath
        self._h5_f = h5py.File(self.h5_fpath, mode='r')
            
        self.normalize_im = normalize_im
        
        self.single_mean_val = single_mean_val
        self.random_translation = random_translation
        
        self.im_transforms = im_transforms
        self.label_transforms = label_transforms
        
        num_data = len(self._h5_f['label'])
        if mask is None:
            self._internal_idx = np.arange(num_data)
        else:
            if num_data != mask.size:
                raise ValueError('Invalid mask!')
            self._internal_idx = np.argwhere(mask).reshape(-1)
            

    def __len__(self):
        return len(self._internal_idx)
    
    def __getitem__(self, idx):
        
        idx = self._internal_idx[idx]
        
        im = self._h5_f["img"][idx, :, :]
        
        if self.normalize_im:
            if self.single_mean_val:
                im = im.astype(np.float32) - self._h5_f["mean"][idx, :, :].mean()
            else:
                im = im.astype(np.float32) - self._h5_f["mean"][idx, :, :]
        
        #im = np.expand_dims(im, 1)
        
        if self.random_translation:
            im = random_translation(im)
            
        #im = im.astype(np.float32)
        im = np.expand_dims(im, -1)
        
        if self.im_transforms is not None:
            im = self.im_transforms(im)
            im = im.float()
            
        label = self._h5_f["label"][idx].astype(np.float)
        if self.label_transforms is not None:
            label = self.label_transforms(label)
            label = label.float()
            
        return im, label
    
    def __del__(self):
        self._h5_f.close()


class PrintScreenDatasetV2H5(Dataset):
    def __init__(self, 
        h5_fpath,
        im_transforms = None,
        label_transforms = None,
        mean_im = None,
    ):
        
        self.h5_fpath = h5_fpath
        self.mean_im = mean_im
        self.im_transforms = im_transforms
        self.label_transforms = label_transforms
            
        self._h5_f = h5py.File(self.h5_fpath, mode='r')
        
    def __len__(self):
        return self._h5_f["label"].shape[0]
    
    def __getitem__(self, idx):
        im = self._h5_f["img"][idx, :, :].astype(np.float)
        im = im.reshape(*im.shape, 1)
        
        if self.im_transforms is not None:
            im = self.im_transforms(im)
            im = im.float()
            
        label = self._h5_f["label"][idx].astype(np.float)
        if self.label_transforms is not None:
            label = self.label_transforms(label)
            label = label.float()
            
        return im, label
    
    def __del__(self):
        self._h5_f.close()
        
class PrintScreenDatasetV3H5(Dataset):
    def __init__(self, 
        h5_fpath,
        im_transforms = None,
        label_transforms = None,
        mean_im = None,
        se_threshold=120,
        num_se_examples=32,
        se_std_val=3,
        num_ref = 10
    ):
        
        self.h5_fpath = h5_fpath
        self.mean_im = mean_im
        self.im_transforms = im_transforms
        self.label_transforms = label_transforms
            
        self._h5_f = h5py.File(self.h5_fpath, mode='r')
        self._orig_dataset_len = self._h5_f["label"].shape[0]
        
        self.se_threshold = se_threshold
        self.num_se_examples = num_se_examples
        self.se_std_val = se_std_val
        self.num_ref = num_ref
        
    def __len__(self):
        return self._orig_dataset_len + 2* self.num_se_examples
    
    def getRef(self):
        refs = self._h5_f["img"][:self.num_ref]  
        refs_list = []

        if self.im_transforms is not None:
            for im in refs:
                ref = self.im_transforms(im)
                ref = ref.float()
                refs_list.append(ref)
                
        refs_tensor = torch.stack(refs_list,dim = 0)
        return refs_tensor
    
    def __getitem__(self, idx):
        if idx < self.num_se_examples:
            im = self._h5_f["img"][idx, :, :].astype(np.float)
            im = im.reshape(*im.shape, 1)
            mask = im <= self.se_threshold
            noise = np.random.randint(self.se_std_val, size=im.shape)
            im[mask] = noise[mask].astype(np.float)
            
            label = self._h5_f["label"][idx].astype(np.float)
            
        elif idx >= self._orig_dataset_len + self.num_se_examples:
            offset = (self._orig_dataset_len + self.num_se_examples)
            new_idx = (self._orig_dataset_len - self.num_se_examples) + (idx - offset)
            
            im = self._h5_f["img"][new_idx, :, :].astype(np.float)
            im = im.reshape(*im.shape, 1)
            mask = im <= self.se_threshold
            noise = np.random.randint(self.se_std_val, size=im.shape)
            im[mask] = self.se_threshold - noise[mask].astype(np.float)
            
            label = self._h5_f["label"][self._orig_dataset_len-1].astype(np.float)
            label += self._h5_f["label"][self.num_se_examples-1].astype(np.float)
            label = float(idx - offset) + 1
            
        else:
            offset = self.num_se_examples
            new_idx = idx - offset     
            
            im = self._h5_f["img"][new_idx, :, :].astype(np.float)
            im = im.reshape(*im.shape, 1)
            
            label = self._h5_f["label"][new_idx].astype(np.float)
            label += self._h5_f["label"][self.num_se_examples-1].astype(np.float)
            label += 1
        
        if self.im_transforms is not None:
            im = self.im_transforms(im)
            im = im.float()
            
        if self.label_transforms is not None:
            label = self.label_transforms(label)
            label = label.float()
            
        return im, label
    
    
    def __del__(self):
        self._h5_f.close()

class DatasetH5TwoRandomGlobalRef(DatasetH5TwoRandom):
    
    def __init__(self,
        h5_fpath,
        num_ref = 10,
        select_num_img = None,      
        transform = None,
        label_transform = None
    ):
        super(DatasetH5TwoRandom,self).__init__()

        self.se_threshold = 120
        self.se_std_val = 3
        
        self.h5_fpath = h5_fpath
        self.num_ref = num_ref
        self.transform = transform
        self.label_transform = label_transform
        

        self._h5_f = h5py.File(self.h5_fpath, mode='r') # the h5 file shape is (2001,720,20)

        self.label = self._h5_f['label'] # label shape: (2001,)

        self.mean = self._h5_f['mean'] # <HDF5 dataset "mean": shape (720, 20)

        self.raw_imgs = self._h5_f['img'] # <HDF5 dataset "img": shape (2001,720, 20)

        self.imgs = np.expand_dims(self.raw_imgs,-1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,720, 20, 1) 
        # self.img = np.expand_dims(self.raw_img,1).astype(np.float) # expand to 4-dims for n-channels: shape (2001,1, 720, 20) 

        self.img_total_num = len(self.label)

        self.all_idx = np.arange(self.img_total_num) # 0-2000 1-D ndarray

    # get first n images as the references.
    def getRef(self):
        refs_list = []
        
        for idx in range(self.num_ref):
            im = self.raw_imgs[idx, :, :].astype(np.float)
            im = im.reshape(*im.shape, 1)
            mask = im <= self.se_threshold
            noise = np.random.randint(self.se_std_val, size=im.shape)
            im[mask] = noise[mask].astype(np.float)

            if self.transform is not None:
                ref = self.transform(im)
                ref = ref.float()
                refs_list.append(ref)
                
        refs_tensor = torch.stack(refs_list,dim = 0)
        return refs_tensor
        
        
        