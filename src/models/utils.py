import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

def findConv2dOutShape(H_in, W_in, conv, pool = 2):
    kernel_size = conv.kernel_size
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    
    H_out=np.floor((H_in+2*padding[0]-dilation[0]*(kernel_size[0]-1)-1)/stride[0]+1)
    W_out=np.floor((W_in+2*padding[1]-dilation[1]*(kernel_size[1]-1)-1)/stride[1]+1)
    
    if pool:
        H_out /= pool
        W_out /= pool
    
    print("height out: " , str(H_out), "Width out : " , str(W_out))
    return int(H_out), int(W_out)

def get_lr(opt):
    for param_group in opt.param_groups:
        return param_group['lr']

# get correct predictions per data batch
def metrics_batch(output, target):
    # get output class
    pred = output.argmax(dim=1, keepdim=True)
    
    # compare output class with target class
    corrects = pred.eq(target.view_as(pred)).sum().item()
    return corrects

def loss_batch(loss_func, output, target, opt=None):
    # get Loss
    loss = loss_func(output, target)
    
    # get performance metric
    metric_b = metrics_batch(output, target)
    
    if opt is not None:
        opt.zero_grad()
        loss.backward()
        opt.step()
        
    return loss.item(), metric_b