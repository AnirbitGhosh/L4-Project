#%%
from pickletools import optimize
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
print(sys.path)

#%%
# Import modules 
from __future__ import print_function, division
from data_loading.data_transform import pretrained_transformer
from data_loading.dataset import breastCancerDataset, show
from data_loading.dataset import show
from models.utils import imshow
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from matplotlib import pyplot as plt
import numpy as np
import time
import copy

# %%
# Load Image training dataset and validation dataset 
data_dir = "D:/PCAM DATA"
data_transforms = pretrained_transformer()

# train_ds = breastCancerDataset(data_dir, data_transforms['train'], "train")
# val_ds = breastCancerDataset(data_dir, data_transforms['validation'], "validation")
img_ds = {
    x: datasets.ImageFolder(os.path.join(data_dir, x+"/grouped_tiles"), 
    data_transforms[x])
    for x in ["train", "validation"]
}

#%%
# Create dataloaders
# dataloaders = {
#     'train' : DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4),
#     'validation' : DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=4)
# }
dataloaders = {
    x: DataLoader(img_ds[x], batch_size=32, shuffle=True)
    for x in ['train', 'validation']
}

# dataset_sizes = {
#     'train' : len(train_ds),
#     'validation' : len(val_ds)
# }
dataset_sizes = {x : len(img_ds[x]) for x in ['train', 'validation']}

# class_names = [0, 1]
class_names = img_ds['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# %%
# create function to train model
def train_pretrained(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    print("starting training epochs...")
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            print(f"in phase: {phase}...")
            if phase == 'train':
                model.train()  # Set model to training mode
                print("Starting model training")
            else:
                model.eval()   # Set model to evaluate mode
                print("Starting model eval")

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                print(f"Iterating in phase: {phase}")
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                print("Setting up no grad...")
                with torch.set_grad_enabled(phase == 'train'):
                    print(f"Getting model outputs in phase: {phase}")
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        print("Starting backpropagation")
                        loss.backward()
                        print("Stepping Optimizer")
                        optimizer.step()

                # statistics
                print("calculating stats...")
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                print("calculating stats...DONE")
            if phase == 'train':
                print("Stepping scheduler")
                scheduler.step()

            print("calculating epoch loss...")
            epoch_loss = running_loss / dataset_sizes[phase]
            print("calculating epoch accuracy...")
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # save best model state dict to path
    path2model = "D:/PCAM DATA/trained_models/weights_02.pt"
    torch.save(best_model_wts, path2model)
    
    return model

# %%
# function to visualize the model
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %%
# Loading a pretrained model and finetuning it for our data
model_pt = models.resnet18(pretrained=True)
num_ftrs = model_pt.fc.in_features

model_pt.fc = nn.Linear(num_ftrs, 2)
model_pt.fc = model_pt.fc.cuda() if torch.cuda.is_available() else model_pt.fc

model_pt = model_pt.to(device)
criterion = nn.CrossEntropyLoss()

optimizer_pt = optim.SGD(model_pt.parameters(), lr=0.0001, momentum=0.9)

# Decay LR by factor 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_pt, step_size=7, gamma=0.1)

# %%
model_pt = train_pretrained(model_pt, criterion, optimizer_pt, exp_lr_scheduler,
                           num_epochs=2)

# %%
visualize_model(model_pt)

# %%