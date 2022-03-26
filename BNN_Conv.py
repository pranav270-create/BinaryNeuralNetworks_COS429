import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import os
import matplotlib
import copy
import tensorflow as tf


# Using torch autograd Function to define derivative w.r.t to binarized weights
# Modifying pytorch implementation from the link below
# https://pytorch.org/docs/stable/autograd.html
class Binarization(torch.autograd.Function):

    @staticmethod
    def forward(ctx,i):
        result = i.sign()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i, = ctx.saved_tensors
        grad_i = grad_output.clone()
        # from Courbariaux et al. paper, need to cancel out gradient in binarized networks if absolute value of weights greater than 1 as it significantly worsens performance otherwise
        grad_i[i.abs()>1.]=0
        #grad_i = tf.clip_by_value(grad_i, clip_value_min=-1, clip_value_max=1)
        return grad_i

    def binarize(input):
        return input.sign()

class BNNLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BNNLinear, self).__init__(*kargs, **kwargs)  
    
    def forward_pass(self, input):
        # check is not the image input size
        if input.size(1) != 784:
            input.data=Binarize(input.data)
        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        self.weight.data=binarize(self.weight.org)
        # from pytorch documentation
        output = F.linear(input, self.weight)
        return output

class BNNConvolution(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BNNConvolution, self).__init__(*kargs, **kwargs)

    def forward_pass(self, input):
        # if not the first image layer, binarize output of previous layer
        if input.size(1) != 3:
            input.data = binarize(input.data)

        if not hasattr(self.weight,'org'):
            self.weight.org=self.weight.data.clone()

        self.weight.data=binarize(self.weight.org)

        # from torch conv2D documentation: https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html
        output = F.conv2d(input, self.weight, None, self.stride, self.padding, self.dilation, self.groups)
        return output

class ConvNet(nn.Module):
    def __init__(self, width = 0.01, norm = 'batch'):
        super(ConvNet, self).__init__()

        self.hidden_layers = 2

        # modeled from example_model_initialization in COS429 Student Assignment 3 - Master
        # define network topology
        conv1 = BNNConvolution(1, 32, kernel_size=5, padding = 2, stride = 2, bias=False)
        norm1 = self.normalization(32,2,norm)
        conv2 = BNNConvolution(32, 64, kernel_size=4, padding = 2, stride =2, bias=False)
        norm2 = self.normalization(64,2,norm)
        linear = BNNLinear(64*64,10, bias=False)
        norm3 = self.normalization(10,1,norm)

        # pytorch ModuleDict https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html
        self.layers = nn.ModuleDict({'cv1': conv1, 'n1': norm1, 'cv2': conv2, 'n2': norm2, 'fc':linear, 'n3':norm3})

        # initialize weights
        for layer in self.layers.keys():
            if not(norm in layer):
                # from pytorch init documentation https://pytorch.org/docs/stable/nn.init.html
                nn.init.normal_(self.layers[layer].weight, mean=0, std=width)
          
    def normalization(self, size, dim, norm):
        # from pytorch documentation: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        if norm == 'batch':
            if dim==2:
                return nn.BatchNorm2d(size, affine=True, track_running_stats=True)
            else:
                return nn.BatchNorm1d(size, affine=True, track_running_stats=True)
        else:
            if dim==2:
                return nn.InstanceNorm2d(size, affine=False, track_running_stats=False)
            else:
                return nn.InstanceNorm1d(size, affine=False, track_running_stats=False)

    def forward(self, x):
        x = self.layers['cv1'](x)
        x = self.layers['n1'](x)
        x = Binarization.apply(x)

        x = self.layers['cv2'](x)
        x = self.layers['n2'](x)
        x = Binarization.apply(x)

        # flatten x before the fully connected layer via pytorch's view function
        x = x.view(x.size(0), -1)
        x = self.layers['fc'](x)
        x = self.layers['n3'](x)

        return x
        
    def save_bn_states(self):
        bn_states = []
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers+1):
                bn = copy.deepcopy(self.layers['bn'+str(l+1)].state_dict())
                bn_states.append(bn)
        return bn_states

    def load_bn_states(self, bn_states):
        if 'bn1' in self.layers.keys():
            for l in range(self.hidden_layers+1):
                self.layers['bn'+str(l+1)].load_state_dict(bn_states[l])
