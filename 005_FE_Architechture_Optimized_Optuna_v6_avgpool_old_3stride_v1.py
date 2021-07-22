import optuna
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset,DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torch_lr_finder import LRFinder

import h5py
from datetime import datetime
import matplotlib.pyplot as plt

from Library.datasets import DatasetH5TwoRandom, DatasetH5ForTest, compute_std_mean
from Library.transforms import RandomFlip,RandomOffset,RandomRotateGrayscale
from Library.scheduler import OneCycleLR,LogLR
from Library.cresnet import initialize_cresnet
from Library.dml import RefBasedDeepMetric
from Library.trainers import RefBasedDeepMetricTrainerV2
'''
005_FE_Architechture_Optimized_Optuna_v6.py
Version 3: 
Trainer: RefBasedDeepMetricTrainerV2
Dataset: Type2 Strid 1; Sequence Index = [2,3,4]; 
mean_val = 0, std_val = 255
avgpool = True
'''

def init_dataset(path,seq_idx_list,mean_val,std_val):

    ds_h5 = []
    refs_list = []

    img_transforms = [RandomRotateGrayscale(rot_range=[-0.1, 0.1], fill=150),
                      RandomFlip(lr_prob=0.5, ud_prob=0.5),
                      transforms.ToTensor(),
                      transforms.Normalize((mean_val,),(std_val,))]
    
    composed_img_transforms = transforms.Compose(img_transforms)
    
    # to obtain the Seq data
    for seq_idx in seq_idx_list:
        h5_fpath = path.format(seq_idx)
        h5_f = DatasetH5TwoRandom(h5_fpath,transform = composed_img_transforms)
        
        # obtain the references, the first 10 images in every Seq.
        refs = h5_f.getRef()
        refs_list.append(refs)
        
        ds_h5.append(h5_f)

    return refs_list, ds_h5

def init_dataloaders(datasets,ds_ratio):
    dataloaders = []
    
    for ds in datasets:
        splitted_ds_num_data = [round(len(ds) * ds_ratio[i]) for i in range(len(ds_ratio))]
        splitted_ds_num_data[0] += len(ds) - int(np.sum(splitted_ds_num_data))

        splitted_ds = []

        for tmp_ds, batch_size, shuffle in zip(random_split(ds, splitted_ds_num_data), batch_sizes, shuffle_flags):

            splitted_ds.append(
                DataLoader(
                    tmp_ds,
                    batch_size=batch_size,
                    shuffle=shuffle,
                    drop_last=True,
                )
            )

        dataloaders.append(splitted_ds)
    return dataloaders

########## Config ##########
## Load Data
# type 2 Data New regenerated data
data_path_type1 = "../MA/NewData/22008_1000202026_652510007"
fnames_type1 = "roi_versuch2_1_{}_720x20.h5"
combined_path = os.path.join(data_path_type1, fnames_type1)
seq_idx_list = [0,1,2,3,4,5,6]
'''
#Type 1 Data
data_path = "../MA/NewData/22008_1000202026_652510007"
fnames = "regenerated_roi_ms1_seq{}_720x30.h5" # 0-8
refs_list, ds_h5 = init_dataset(data_path,fnames)
'''
# Dataloader for training set ,validation set, test set (80%,20%)
inputs_shape = [1,720,20]
batch_sizes = [32, 32]
shuffle_flags = [True, False]
ds_ratio = [0.8, 0.2]

# Initialize datasets 
#mean_val, std_val = compute_std_mean(seq_idx_list, combined_path)
mean_val = 0
std_val = 255
refs_list, ds_h5 = init_dataset(combined_path, seq_idx_list, mean_val, std_val)

# Initialize dataloaders
dataloaders = init_dataloaders(ds_h5, ds_ratio)
train_dls = [dls[0] for dls in dataloaders]
valid_dls = [dls[1] for dls in dataloaders]

# Training config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
num_epochs = 50
phases_ratio = [0.2, 0.2, 0.4, 0.2]

### EXP:
exp_num = 0

# full_strides = [[2,1], 2, [2,1], 2, [2,1], 2, [2,1], 2, [2,1]]
n_channels_template = np.array([8, 8, 16, 16, 32, 32, 64, 64, 128], dtype=int)
def objective(trial):
    
    n_groups = trial.suggest_int('n_groups', 2, 9) # number of layers, each layer contains some blocks
    n_blocks = np.ones(n_groups, dtype=int) * trial.suggest_int('x_blocks', 1, 3)
    #n_channels = n_channels_template[:n_groups] * trial.suggest_int('x_channels', 1, 4)
    n_channels = n_channels_template[:n_groups]

    strides = []
    for i_group in range(n_groups):
        
        stride_r_val = 2 # set to 2
        if i_group < 2:
            stride_c_val = 2
        else:
            stride_c_val = trial.suggest_int('stride_c{}'.format(i_group), 1, 2)
        
        strides.append([stride_r_val, stride_c_val])
    
    #block_name = trial.suggest_categorical('block_name', ['BasicBlock', 'LeakyBasicBlock'])
    block_name = trial.suggest_categorical('block_name', ['bottleneck', 'leakybottleneck'])

    #enable_avgpool = trial.suggest_categorical('enable_avgpool', [True, False])
    
    n_neurons_val0 = trial.suggest_int('n_neurons_val0', 5, 8)
    n_neurons_val1 = trial.suggest_int('n_neurons_val1', 4, n_neurons_val0-1)
    
    n_neurons = [
        2**n_neurons_val0, 
        #2**n_neurons_val1,
    ]

    cresnet_config = {
        'InputShape' : inputs_shape,
        'BlockName' : block_name,
        'NumBlocks' : n_blocks,
        'NumChannels' : n_channels,
        'Strides' : strides,
        'NumNeurons' : n_neurons,
        'EnableGlobalAveragePooling' : True,
        'PrintIntermediateShape' : True,
        'InitializeResidual': True,
    }

    print(cresnet_config)

    # To initialize custom resnet as feature extractor
    try:
        feature_extractor = initialize_cresnet(cresnet_config, device)
        model = RefBasedDeepMetric(
            feature_extractor, 
            loss_non_linearity_name='sigmoid',
            criterion_name='bce',
        )
    except RuntimeError:
        raise optuna.exceptions.TrialPruned()
        score = np.nan
        
    optim_f = torch.optim.Adam
    lr_val = trial.suggest_loguniform('lr_val', 1e-6, 1e-2)
    optim_kwargs = {'lr' : lr_val}  
    
    scheduler_f = OneCycleLR
    min_lr_factor_val = trial.suggest_int('min_lr_factor_val', 1, 10)
    annealing_lr_factor_val = trial.suggest_int('annealing_lr_factor_val', 1, 10)
    
    scheduler_kwargs = {
        'min_lr_factor' : 5e-2*min_lr_factor_val, 
        'annealing_lr_factor' : 5e-2*annealing_lr_factor_val,
        'phases_ratio' : [0.2, 0.2, 0.4, 0.2]
    }
    print()
    print(cresnet_config)
    print('LR: {}'.format(lr_val))
    print(scheduler_kwargs)
    print()
    
    global exp_num
    result_path = os.path.join("{}_results".format(study_name), "{:03d}".format(exp_num))
    exp_num += 1
    
    #try:
    trainer = RefBasedDeepMetricTrainerV2(
        train_dls,
        valid_dls,
        refs_list,
        batch_sizes,
        num_epochs,
        device,
        model,
        optim_f,
        optim_kwargs,
        scheduler_f,
        scheduler_kwargs,
        result_path=result_path,
        enab_summary_writer = True,
        enab_tqdm = True,
        per_seq_valid = True
    )
    score = trainer.run_optuna()

    #except RuntimeError:
        #score = np.nan
        #raise optuna.exceptions.TrialPruned()
        
    return score


study_name = 'NN_Optimized_Optuna_v6_old_3stride'  # Unique identifier of the study.
study = optuna.create_study(study_name=study_name, storage='sqlite:///{}.db'.format(study_name), load_if_exists=True)
study_df = study.trials_dataframe(attrs=('number', 'value', 'params', 'state'))
exp_num = len(study_df)
print("Starting exp num:{}".format(exp_num))
study.optimize(objective, n_trials=300)

pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

print("Study statistics: ")
print(" Number of finished trials: ", len(study.trials))
print(" Number of pruned trials: ", len(pruned_trials))
print(" Number of complete trials: ", len(complete_trials))

print("Best Trial: ")
trial = study.best_trial
print(" Value: ", trial.value)

print(" Params: ")
for key, value in trial.params.items():
    print(" {}: {}".format(key,value))