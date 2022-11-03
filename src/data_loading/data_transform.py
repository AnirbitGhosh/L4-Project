import torchvision.transforms as transforms 
import numpy as np
 #transforms.RandomResizedCrop(96, scale=(0.8, 1.0), ratio=(1.0,1.0)),
       
def train_transformer():
    train_transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])
    
    return train_transformer

def validation_transfomer():
    val_transformer = transforms.Compose([transforms.ToTensor()])
    return val_transformer

def pretrained_transformer():
    transformer = {
        'train' : transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ToTensor()
        ]),
        'validation' : transforms.Compose([
            transforms.ToTensor()
        ])
    }
    
    return transformer

def accuracy(labels, out):
    return np.sum(out==labels)/float(len(labels))
    