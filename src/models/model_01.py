#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
print(sys.path)

# %%
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

# %%
# set base paths
base_dir_path = "D:/PCAM DATA"

train_transformer = train_transformer()
val_transformer = validation_transfomer()

# %%
# create training & validation Dataset using custom Dataset Model
#bc_dataset = breastCancerDataset(base_dir_path, train_transformer, "train")

# len_ds = len(bc_dataset)
# len_train = int(0.8*len_ds)
# len_val = len_ds - len_train

train_ds = breastCancerDataset(base_dir_path, train_transformer, "train")
val_ds = breastCancerDataset(base_dir_path, val_transformer, "validation")

# %%
# create Dataloaders
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=64, shuffle=False)

# %%
# Extract training batch
for x, y in train_dl:
    print(x.shape)
    print(y.shape)
    break

# %%
# Get a data batch from validation DL
for x, y in val_dl:
    print(x.shape)
    print(y.shape)
    break

#%%
# get labels for val dataset
y_val = [y for _,y in val_ds]

# %%
# dumb baseline for all-false predictions
acc_all_false = accuracy(y_val, np.zeros_like(y_val))
print("accuracy all false prediction: %.2f" %acc_all_false)

# %%
# dumb baseline for all-true predictions
acc_all_true = accuracy(y_val, np.ones_like(y_val))
print("accuracy all true prediction: %.2f" %acc_all_true)

# %%
# dumb baseline for random predictions
acc_random = accuracy(y_val, np.random.randint(2,size=len(y_val)))
print("accuracy all random prediction: %.2f" %acc_random)

# %%
# example output shape prediction
conv1 = nn.Conv2d(3, 8, kernel_size=3)
h,w = findConv2dOutShape(96, 96, conv1)
print(h, w)

# %%
# dict to define custom Net parameters
params_model = {
    "input_shape" : (3, 96, 96),
    "initial_filters" : 8,
    "num_fc1" : 100,
    "dropout_rate" : 0.25,
    "num_classes" : 2
}

# %%
# create Neural Network model
cnn_model = Net(params_model)

# %%
print("Is CUDA available: ", torch.cuda.is_available())

# %%
# move model to GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    cnn_model = cnn_model.to(device)
    
# %%
# model parameters
print(cnn_model.parameters)

# %%
# model device check
print("Device: ", next(cnn_model.parameters()).device)

# %%
# get custom model summary
summary(cnn_model, input_size=(3, 96, 96), device=torch.device('cuda').type)

# %%
# define loss function
loss_func = nn.NLLLoss(reduction="sum")

# %%
# loss func example
n, c = 8, 2
y = torch.randn(n, c, requires_grad=True)
ls_F = nn.LogSoftmax(dim=1)
y_out=ls_F(y)
print(y_out.shape)

target = torch.randint(c, size=(n,))
print(target.shape)

loss = loss_func(y_out, target)
print(loss.item())

loss.backward()
print(y.data)

# %%
# define an ADAM optimizer with learning rate of 3e-4
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)

# %%
# read current value of learning reate using get_lr func defined in utils
curr_lr = get_lr(opt)
print('current lr = {}'.format(curr_lr))

# %%
# define learning rate scheduler 
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=20, verbose=1)

# %%
# test lr scheduler
for i in range(100):
    lr_scheduler.step(i)
    
# %%
## Training and Evaluation

loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=20, verbose=1)

params_train={
    "num_epochs": 100,
    "optimizer": opt,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": True,
    "lr_scheduler": lr_scheduler,
    "path2weights": "D:/PCAM DATA/trained_models/weights_01_10k.pt",
}

# %%
# train and validate the model
cnn_model, loss_hist, metric_hist = train_val(cnn_model, params_train)

# %%
# Train-Validation progress
num_epochs = params_train["num_epochs"]

# plot loss progress
plt.title("Train-Val loss")
plt.plot(range(1, num_epochs+1), loss_hist["train"], label="train")
plt.plot(range(1, num_epochs+1), loss_hist["val"], label="val")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend()
plt.show()

# %%
# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# %%
loss_func = nn.NLLLoss(reduction="sum")
opt = optim.Adam(cnn_model.parameters(), lr=3e-4)
lr_scheduler = ReduceLROnPlateau(opt, mode='min',factor=0.5, patience=20,verbose=1)

params_train={
 "num_epochs": 2,
 "optimizer": opt,
 "loss_func": loss_func,
 "train_dl": train_dl,
 "val_dl": val_dl,
 "sanity_check": False,
 "lr_scheduler": lr_scheduler,
 "path2weights": "D:/PCAM DATA/trained_models/weights_01_10k.pt",
}

# train and validate the model
cnn_model,loss_hist,metric_hist=train_val(cnn_model,params_train)

# %%
# Train-Validation Progress
num_epochs=params_train["num_epochs"]

# plot loss progress
plt.title("Train-Val Loss")
plt.plot(range(1,num_epochs+1),loss_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),loss_hist["val"],label="val")
plt.ylabel("Loss")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()

# plot accuracy progress
plt.title("Train-Val Accuracy")
plt.plot(range(1,num_epochs+1),metric_hist["train"],label="train")
plt.plot(range(1,num_epochs+1),metric_hist["val"],label="val")
plt.ylabel("Accuracy")
plt.xlabel("Training Epochs")
plt.legend()
plt.show()
# %%
