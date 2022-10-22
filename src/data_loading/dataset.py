import torch
from torch.utils.data import Dataset
import pandas as pd
import torchvision.transforms as transforms
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

torch.manual_seed(0)
class breastCancerDataset(Dataset):
    def __init__(self, data_dir, transform, data_type='train'):
        
        # path to images
        data_path = os.path.join(data_dir, data_type+"/tiles")
        
        # get list of images
        fnames = os.listdir(data_path)
        
        # get full path to images
        self.full_fnames = [os.path.join(data_path, f) for f in fnames]
        
        # labels are in a csv file names train_labels.csv
        labels_path = os.path.join(data_dir, data_type+"/" + data_type + "_labels.csv")
        labels_df = pd.read_csv(labels_path)
        
        # set data frame index to id
        labels_df.set_index("id", inplace=True)
        
        # obtain labels from data frame
        self.labels = [labels_df.loc[int(f[:-5])].values[3] for f in fnames]
        
        self.transform = transform
        
    def __len__(self):
        return len(self.full_fnames)
    
    def __getitem__(self, index):
        # open image, apply transform and return with label
        image = Image.open(self.full_fnames[index])
        image = self.transform(image)
        return image, self.labels[index]
    

def show(img, y, color=False):
    npimg = img.numpy()
    
    # convert to H*W*C shape
    npimg_tr = np.transpose(npimg, (1,2,0))
    
    if not color:
        npimg_tr = npimg_tr[:,:,0]
        plt.imshow(npimg_tr, cmap='gray', interpolation='nearest')
    else: 
        plt.imshow(npimg_tr, interpolation='nearest')
    plt.title("label: " + str(y))
