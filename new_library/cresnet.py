import numpy as np

import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as f

from .resnet import conv1x1, conv3x3, BasicBlock, Bottleneck, LeakyBasicBlock, LeakyBottleneck

class CResNet(nn.Module):

    def __init__(self, 
        input_shape, 
        block, 
        nblocks, 
        nchannels, 
        strides,
        nneurons,
        ena_glob_aver_pool=False,
        zero_init_residual=False,
        print_output_shape=False,
    ):
        
        super(CResNet, self).__init__()
        
        if nneurons is None or len(nneurons) == 0:
            self.any_fc = False
        else:
            self.any_fc = True
            
        self.ena_glob_aver_pool = ena_glob_aver_pool
        
        self.nchannels = nchannels
        self.inplanes = input_shape[0]
        
        if len(nblocks) != len(nchannels) or len(nchannels) != len(strides):
            raise ValueError('Invalid nblocksor nchannels or strides')
            
        self.num_cnn = len(nblocks)
        
        for i in range(self.num_cnn):
            setattr(
                self, 
                "Block{}".format(i),
                self._make_layer(block, self.nchannels[i], nblocks[i], stride=strides[i])
            )
        
        test_output = torch.ones(1, *input_shape)
        
        self.intermediate_output_shape = []
        for i in range(self.num_cnn):
            test_output = getattr(self, "Block{}".format(i))(test_output)
            self.intermediate_output_shape.append(
                test_output.shape
            )
            
        if self.ena_glob_aver_pool:
            self.aver_pool = nn.AdaptiveAvgPool2d((1,1))
            
            test_output = self.aver_pool(test_output)
            self.intermediate_output_shape.append(
                test_output.shape
            )
        
        if self.any_fc:
            nneurons.insert(0, np.product(test_output.shape[1:]))
            self.nneurons = nneurons

            self.num_fc = len(self.nneurons)-1

            lin_layers = []
            for i in range(self.num_fc-1):
                setattr(
                    self, 
                    "FClayer{}".format(i),
                    nn.Sequential(
                        nn.Linear(nneurons[i], nneurons[i+1]),
                        nn.ReLU(inplace=True)
                    )
                )

            i = self.num_fc-1
            setattr(
                self, 
                "FClayer{}".format(i),
                nn.Sequential(
                    nn.Linear(nneurons[i], nneurons[i+1]),
                )
            )
            
        if print_output_shape:
            print(self.intermediate_output_shape)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, out_planes=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward_conv(self, x):
        for i in range(self.num_cnn):
            x = getattr(self, "Block{}".format(i))(x)
        
        return x
    
    def forward_fc(self, x):
        for i in range(self.num_fc):
            x = getattr(self, "FClayer{}".format(i))(x)
        
        return x

    def forward(self, x):

        x = self.forward_conv(x)
        
        if self.ena_glob_aver_pool:
            x = self.aver_pool(x)
        
        if self.any_fc:
            x = x.view(x.size(0), -1)    
            x = self.forward_fc(x)

            x = x.view(x.size(0))

        return x
    
def initialize_cresnet(config, device, weight_fpath=None):
    
    input_shape = config['InputShape']
    block_name = config['BlockName']
    nblocks = config['NumBlocks']
    nchannels = config['NumChannels']
    strides = config['Strides']
    nneurons = config.get('NumNeurons', None)
    ena_glob_aver_pool = config.get('EnableGlobalAveragePooling', False)
    
    print_output_shape = config.get('PrintIntermediateShape', False)
    
    if block_name.lower() == 'basicblock':
        block = BasicBlock
    elif block_name.lower() == 'bottleneck':
        block = Bottleneck
    elif block_name.lower() == 'leakybasicblock':
        block = LeakyBottleneck
    elif block_name.lower() == 'leakybottleneck':
        block = LeakyBottleneck
    else:
        raise NotImplementedError("{} is not implemented".format(block_name))
                                  
    zero_init_residual = config.get('InitializeResidual', True)
        
    model = CResNet(
        input_shape, 
        block, 
        nblocks, 
        nchannels, 
        strides,
        nneurons,
        ena_glob_aver_pool=ena_glob_aver_pool,
        zero_init_residual=zero_init_residual,
        print_output_shape=print_output_shape,
    )

    model.to(device)
                    
    if weight_fpath:
        model.load_state_dict(
            torch.load(weight_fpath)
        )
    
    return model