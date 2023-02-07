import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))

# import modules
from data_loading.dataset import breastCancerDataset
from data_loading.dataset import show
from data_loading.data_transform import accuracy, train_transformer, validation_transfomer
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
import numpy as np
from custom_network import Net, train_val
from utils import findConv2dOutShape, get_lr
from matplotlib import pyplot as plt
import copy
import argparse

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=dir_path, help='Pass base directory path name with -p or --path flags')

def build_datasets(dir):
    train_transform = train_transformer()
    val_transform = validation_transfomer()
    
    train_ds = breastCancerDataset(dir, train_transform, "train")
    val_ds = breastCancerDataset(dir, val_transform, "validation")
    
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)
    
    return train_dl, val_dl

def create_model(network, model_params):
    cnn = network(model_params)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        cnn = cnn.to(device)

    return cnn
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        parser.print_usage()
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    print("Parsing arguments...")
    base_dir = args.path
    
    print("Building datasets....")
    train, validation = build_datasets(base_dir)
    
    print("Setting up CNN Model using params: ")
    params_model = {
        "input_shape" : (3, 96, 96),
        "initial_filters" : 8,
        "num_fc1" : 100,
        "dropout_rate" : 0.25,
        "num_classes" : 2,
        "activation_func" : 'tanh',
    }
    print(params_model)
    model = create_model(Net, params_model)
    
    
    print("\nsetting up training params...")
    loss_func = nn.NLLLoss(reduction="sum")
    opt = optim.Adam(model.parameters(), lr=3e-4)
    # opt = optim.SGD(model.parameters(), lr=3e-4)
    lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=10, verbose=1)

    
    params_train={
        "num_epochs": 100,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train,
        "val_dl": validation,
        "sanity_check": True,
        "lr_scheduler": lr_scheduler,
        "save_weights": False,
        "path2weights": "D:/PCAM DATA/trained_models/weights_01_100k_normalized.pt",
    }
    print(params_train)
    
    print("\nStarting model training...")
    cnn_model, loss_hist, metric_hist = train_val(model, params_train)
    print("\nModel trained and saved.")