#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from custom_network import Net, train_val
from utils import findConv2dOutShape, get_lr
from model_01 import build_datasets
from genericpath import isdir
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import argparse

torch.manual_seed(0)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--base', type=dir_path, help='Pass base directory path with -d flag', default="D:/PCAM DATA")

class TuningDataset(Dataset):
    def __init__(self, data_dir, transform, data_type='train'):
        
        # path to images
        data_path = os.path.join(data_dir, data_type+"/tiles")
        
        # get list of images
        fnames = os.listdir(data_path)
        
        self.full_fnames = [os.path.join(data_path, f) for f in fnames][:10000]
        
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
        if isinstance(index, slice):
            start = 0 if index.start == None else index.start
            stop = -1 if index.stop == None else index.stop
            step = 1 if index.step == None else index.step
            images = []
            for idx in range(start, stop, step):
                img = Image.open(self.full_fnames[idx])
                img = self.transform(img)
                images.append(img)
            return images, self.labels[start:stop:step]
        else:
            # open image, apply transform and return with label
            image = Image.open(self.full_fnames[index])
            image = self.transform(image)
            return image, self.labels[index]

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

def build_dataset_tune(dir, train_batch_size, val_batch_size):
    train_transform = train_transformer()
    val_transform = validation_transfomer()
    
    train_ds = TuningDataset(dir, train_transform, "train")
    val_ds = TuningDataset(dir, val_transform, "validation")
    
    train_dl = DataLoader(train_ds, batch_size=train_batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False)
    
    return train_dl, val_dl

def load_test_data(dir):
    val_transform = validation_transfomer()
    
    test_ds = TuningDataset(dir, val_transform, "test")
    
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    return test_dl

def train_cifar(config, base_dir, checkpoint_dir=None, data_dir=None):
    model_params = {
        "input_shape" : (3, 96, 96),
        "initial_filters" : 8,
        "num_fc1" : 100,
        "dropout_rate" : config["dropout_rate"],
        "num_classes" : 2,
        "activation_func" : config["activation_func"],
    }
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'
    
    net = Net(model_params)
    net.to(device)
    
    loss_func = nn.NLLLoss(reduction="sum")
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])
    lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=1)
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    train_dl, val_dl = build_dataset_tune(base_dir, config['train_batch'], config['val_batch'])
    
    train_params = {
        "num_epochs": 100,
        "optimizer": optimizer,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "sanity_check": True,
        "lr_scheduler": lr_scheduler,
    }
    
    for epoch in range(100):
        running_loss = 0.0
        epoch_steps = 0
        print(f"Training Epoch {epoch+1} : ")
        
        for i, data in enumerate(train_dl, 0):
            if i%200 == 0:
                print(f"train batch: {i}")
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            epoch_steps += 1
            if i%500 == 0:
                print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
            torch.cuda.empty_cache()
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_dl, 0):
            if i%200 == 0:
                print(f"valid batch: {i}")
            with torch.no_grad():
                val_inputs, val_labels = data
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                
                val_output = net(val_inputs)
                predicted = val_output.argmax(dim=1, keepdim=True)
                corrects = predicted.eq(val_labels.view_as(predicted)).sum().item()
                total += val_labels.size(0)
                correct += corrects
                
                loss = loss_func(val_output, val_labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((net.state_dict(), loss_func.state_dict()), path)
        
        
        tune.report(loss=(val_loss / val_steps), accuracy=correct/total)
    print("Finished training")

def test_accuracy(net, base_dir, device="cpu"):
    tests_dl = load_test_data(base_dir)
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tests_dl:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total

 
def main(base_dir, num_samples=10, max_num_epochs=100):
    config = {
        "activation_func": tune.choice(['tanh', 'relu', 'leaky relu']),
        "dropout_rate": tune.uniform(0, 1),
        "train_batch": tune.choice([32, 64, 128]),
        "val_batch" : tune.choice([32, 64, 128]),
        "lr": tune.loguniform(1e-5, 1e-1)
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=10,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "accuracy", "training_iteration"]
    )
    result = tune.run(
        partial(train_cifar, base_dir),
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.join(base_dir, "tuning"),
        resources_per_trial={"gpu": 1, "cpu": 5}),
    
    best_trial = result[0].get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    
    best_model_params = {
        "input_shape" : (3, 96, 96),
        "initial_filters" : 8,
        "num_fc1" : 100,
        "dropout_rate" : best_trial.config["dropout_rate"],
        "num_classes" : 2,
        "activation_func" : best_trial.config["activation_func"],
    }
    best_trained_model = Net(best_model_params)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.dir_or_data
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)
    
    test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(test_acc))
    
#%%
if __name__ == "__main__":
    args = parser.parse_args()
    
    main(args.base, num_samples=20, max_num_epochs=50)
