#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
from data_loading.data_transform import validation_transfomer, pretrained_pred_transformer
from models.custom_network import Net
from pretrained_model.pretrained_network import PretrainedNet
import torch
import matplotlib.pyplot as plt
from torchvision import models
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

#%%
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
def file_path(string):
    if os.path.isfile(string):
        return string
    else:
        raise FileNotFoundError(string)
    
#%%
parser = argparse.ArgumentParser()
parser.add_argument('-n', '--net', type=file_path, help='Pass path to file containing model weights with -i or --image flags' )
parser.add_argument('-t', '--tiles', type=dir_path, help="Pass directory containing tiles to predict lables of using -l or --label flags")
parser.add_argument('-o', '--output', type=dir_path, help="Pass output directory using -o or --output flags")

device = "cuda:0" if torch.cuda.is_available() else 'cpu'

# %%
def read_model(weights):
    params_model = {
        "input_shape" : (3, 96,96),
        "initial_filters" : 8,
        "num_fc1" : 100,
        "dropout_rate" : 0.25,
        "num_classes" : 2
    }
    
    model = Net(params_model)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint)
    
    return model

def get_predictions(model, tile_dir, image_dir, output_dir):
    transform  = validation_transfomer()
    
    print("Getting each whole slide image name...")
    for file in os.listdir(image_dir):
        dirname = file[:-4]
        print(f"Processing predictions for image : {dirname}")
        tile_path = os.path.join(tile_dir, dirname)
    
        predictions = []
        print("Starting prediction generation for tiles...")
        for image_file in os.listdir(tile_path):
            image_name = image_file[:-4]
            image = Image.open(os.path.join(tile_path, image_file))
            print("Predicting class for image {} ...".format(image_name))
            input = transform(image)
            
            output = model(input)
            prediction = int(torch.max(output.data, 1)[1].numpy())
            print("Prediction success - saving output!")
            predictions.append(prediction)

        df = pd.DataFrame({"image" : os.listdir(tile_path), "predictions" : predictions})
        csv_name = dirname + "-predictions-normalized.csv"
        df.to_csv(os.path.join(output_dir, csv_name))     
        
#%%
if __name__ == "__main__" :
    if len(sys.argv) < 6:
        parser.print_usage()
        parser.print_help()
        sys.exit(1)
        
    print("Parsing arguments...")
    args = parser.parse_args()
    print("Parsing arguments... DONE")
    
    print("Locating directories...")
    net_path = args.net
    tile_dir = args.tiles
    out_dir = args.output
    image_dir = "D:/PCAM DATA/WSI/Whole Slide Images"
    print("Locating directories... DONE!")
    
    print("Generating model with given weights... ")
    model = read_model(net_path)
    print("Generating model with given weights... DONE!")
    
    get_predictions(model=model, tile_dir=tile_dir, image_dir=image_dir, output_dir=out_dir)
    # get_pretrained_predictions(model=model, tile_dir=tile_dir, image_dir=image_dir, output_dir=out_dir)
