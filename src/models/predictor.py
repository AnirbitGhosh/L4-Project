#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
print(sys.path)

# %%
from data_loading.data_transform import validation_transfomer
from models.custom_network import Net
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='Pass path to model to evaluate with -m flag')
parser.add_argument('-o', '--output', help="Pass desired output path with filename with -o flag", default="D:/PCAM DATA/WSI/Tiles/TCGA-Slide-01-predictions.csv")

# %%
if __name__ == "__main__" :
    
    args = parser.parse_args()
    model_path = args.model
    output_path = args.output
    
    
    params_model = {
        "input_shape" : (3, 96,96),
        "initial_filters" : 8,
        "num_fc1" : 100,
        "dropout_rate" : 0.25,
        "num_classes" : 2
    }

    model = Net(params_model)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    # %%
    # load input from dir and get predictions for each
    transform  = validation_transfomer()

    predictions = []
    image_dir = "D:\PCAM DATA\WSI\Tiles\TCGA-Slide-01"
    for file in os.listdir(image_dir):
        image = Image.open(image_dir + "/" + file)
        print("Predicting class for image {} ...".format(file))
        input = transform(image)
        
        output = model(input)
        prediction = int(torch.max(output.data, 1)[1].numpy())
        print("Prediction success - saving output!")
        predictions.append(prediction)


    #%%
    df = pd.DataFrame({"image": os.listdir(image_dir), "predictions":predictions})
    df.head()
    df.to_csv(output_path)

    # %%
    # pass input image to model and print prediction

    # output = model(input)
    # prediction = int(torch.max(output.data, 1)[1].numpy())

    # output = model(input)
    # prediction = int(torch.max(output.data, 1)[1].numpy())
    # print("Model prediction: ", prediction)
    # if prediction == 1:
    #     print("The tile is malignant: ", True)
    # else: 
    #     print("The tile is malignant:", False)
        
    #%%
    images= [plt.imread(image_dir + "/" + img) for img in os.listdir(image_dir)[:5000]]
    print(len(images))

    plt.rcParams['figure.figsize'] = (10,10)
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    nrows, ncols = 10, 10

    idx = 0
    for img, prediction in zip(images[4500:4600], predictions[4500:4600]):
        
        plt.subplot(nrows, ncols, idx+1)
        plt.axis('off')
        if prediction == 1:
            disp = img[:,:,0]
            plt.imshow(disp, cmap='Reds')
            # plt.title("MALIGNANT : TRUE")
        else:
            plt.imshow(img)
            # plt.title("MALIGNANT : FALSE")
        idx+=1

    # %%
