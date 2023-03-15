#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
print(sys.path)

from data_loading.data_transform import validation_transfomer, train_transformer
from data_loading.dataset import breastCancerDataset
from models.model_01 import build_datasets, create_model
from models.custom_network import Net, train_val
import torch
import torch.nn as nn
import torch
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn import metrics
import seaborn as sns
import argparse

#%%
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', help='Pass path to model to evaluate with -m flag')
parser.add_argument('-t', '--test', type=dir_path, help="Pass test directory path with -d flag")

#%%
def read_model(weights):
    params_model = {
        "input_shape" : (3, 96, 96),
        "initial_filters" : 8,
        "num_fc1" : 100,
        "dropout_rate" : 0.25,
        "num_classes" : 2,
        "activation_func" : 'tanh',
    }
    
    model = Net(params_model)
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint)
    
    return model

def generate_predictions(model, images):
    transform = validation_transfomer()
    predictions = []
    
    for file in os.listdir(images):
        image = Image.open(images + "/" + file)
        input = transform(image)
        output = model(input)
        prediction = int(torch.max(output.data, 1)[1].numpy())
        predictions.append(prediction)
    
    return predictions

def calc_accuracy(df):
    corrects = 0
    for id in df.index:
        predicted = df.loc[id]['prediction']
        actual = df.loc[id]['tumor_patch']
        if predicted == actual:
            corrects += 1
    
    return corrects/len(df)

def calc_recall(df):
    tp = 0
    fn = 0
    for id in df.index:
        predicted = df.loc[id]['prediction']
        actual = df.loc[id]['tumor_patch']
        
        if actual == 1 and predicted == actual:
            tp += 1
        if actual == 1 and predicted != actual:
            fn += 1

    return (tp)/(tp + fn)

def calc_precision(df):
    tp = 0
    fp = 0 
    for id in df.index:
        predicted = df.loc[id]['prediction']
        actual = df.loc[id]['tumor_patch']
        
        if actual == 1 and actual == predicted:
            tp += 1
        if actual == 0 and actual != predicted:
            fp += 1
            
    return (tp)/(tp + fp)

#%%
if __name__ == "__main__":
    args = parser.parse_args()
    
    model = read_model(args.model)
    test_image_dir = os.path.join(args.base, "tiles")
    test_ground_truth = os.path.join(args.test, "test_labels.csv")

    test_results = generate_predictions(model, test_image_dir)

    # %%
    label_df = pd.read_csv(test_ground_truth)
    label_df = label_df[['id', 'tumor_patch']].astype(int)

    label_df = label_df.head(20000)
    label_df['prediction'] = np.array(test_results)
    label_df.head()

    # %%
    accuracy = calc_accuracy(label_df)
    print(f"accuracy : {accuracy}")

    recall = calc_recall(label_df)
    print(f"recall : {recall}")

    precision = calc_precision(label_df)
    print(f"precision : {precision}")

    # %%
    confusion_matrix = metrics.confusion_matrix(label_df['tumor_patch'], label_df['prediction'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["Benign", "Malignant"])
    cm_display.plot(cmap="Blues")
    plt.show()

    # %%
    ax = plt.subplot()
    sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap=sns.color_palette("rocket_r", as_cmap=True))
    ax.xaxis.set_ticklabels(['Benign', 'Malignant'])
    ax.yaxis.set_ticklabels(['Benign', 'Malignant'])
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
