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

# %%
params_model = {
    "input_shape" : (3, 96,96),
    "initial_filters" : 8,
    "num_fc1" : 100,
    "dropout_rate" : 0.25,
    "num_classes" : 2
}

model = Net(params_model)
checkpoint = torch.load('D:/PCAM DATA/trained_models/weights_01.pt')
model.load_state_dict(checkpoint)

# %%
# load input from dir and get predictions for each
transform  = validation_transfomer()

predictions = []
image_dir = "C:/Users/Anirbit/L4 Project/src - practice/WSI-processing/WSI-data/Tiles/TCGA-slide-01"
for file in os.listdir(image_dir):
    image = Image.open(image_dir + "/" + file)
    print("Predicting class for image {} ...".format(file))
    input = transform(image)
    
    output = model(input)
    prediction = int(torch.max(output.data, 1)[1].numpy())
    print("Prediction success - saving output!")
    predictions.append(prediction)

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
images= [plt.imread(image_dir + "/" + img) for img in os.listdir(image_dir)[2200:2300]]
print(len(images))

plt.rcParams['figure.figsize'] = (10,10)
plt.subplots_adjust(wspace=0.2, hspace=0.2)
nrows, ncols = 10, 10

idx = 0
for img, prediction in zip(images[:100], predictions[:100]):
    
    plt.subplot(nrows, ncols, idx+1)
    plt.axis('off')
    if prediction == 1:
        disp = img
        plt.imshow(disp)
        # plt.title("MALIGNANT : TRUE")
    else:
        plt.imshow(img[:,:, 1], cmap='gray')
        # plt.title("MALIGNANT : FALSE")
    idx+=1

# %%
