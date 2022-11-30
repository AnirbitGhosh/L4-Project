import numpy as np
import torch 
from visdom import Visdom 

def metrics(prediction, target):
    prediction_binary = torch.ge(prediction, 0.5).float()
    N = target.numel()
    
    # determining True positives, true negatives, false positives, false negatives
    true_pos = torch.nonzero(prediction_binary * target).shape[0]
    true_neg = torch.nonzero((1-prediction_binary) * (1-target)).shape[0]
    false_pos = torch.nonzero(prediction_binary * (1 - target)).shape[0]
    false_neg = torch.nonzero((1-prediction_binary) * target).shape[0]
    
    accuracy = (true_pos + true_neg) /N
    precision = 0. if true_pos == 0 else true_pos / (true_pos + false_pos)
    recall = 0. if true_pos == 0 else true_pos / (true_pos + false_neg)
    specificity = 0. if true_neg == 0 else true_neg / (true_neg + false_pos)
    f1 = 0. if precision == 0 or recall == 0 else (2 * precision * recall) / (precision + recall)
    
    return {"accuracy" : accuracy, "precision" : precision, "recall" : recall,  "f1" : f1, "specificity" : specificity}

def loss(prediction, target):

    w1 = 1.33  # False negative penalty
    w2 = .66  # False positive penalty

    return -torch.mean(w1 * target * torch.log(prediction.clamp_min(1e-3))
                       + w2 * (1. - target) * torch.log(1. - prediction.clamp_max(.999)))
class Writer(object):
    def __init__(self, env_name="main"):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}
        
    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title= title_name,
                xlabel="Iterations",
                ylabel=var_name,
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update="append")
        
def sliding_window(image_shape, window_shape, stride=None):
    if stride is None:
        stride = (window_shape[0], window_shape[1])
    
    padding_x = 0 if image_shape[1] % window_shape[1] == 0 else window_shape[1] - image_shape[1] % window_shape[1]
    padding_y = 0 if image_shape[0] % window_shape[0] == 0 else window_shape[0] - image_shape[0] % window_shape[0]
    padded_shape = (image_shape[0]+padding_y, image_shape[1] + padding_x)
    
    x = np.arange(0, padded_shape[1], stride[1])
    y = np.arange(0, padded_shape[0], stride[0])
    
    x1, y1, = np.meshgrid(x, y)
    x2 = x1 + window_shape[1]
    y2 = y1 + window_shape[0]
    
    return np.stack([x1, y1, x2, y2], axis=2), {'x' : padding_x, 'y' : padding_y}