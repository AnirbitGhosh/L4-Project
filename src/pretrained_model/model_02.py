import argparse
import numpy as np
import torch
import torch.optim as optim

import pretrained_network
import data_loader
import utils

parser = argparse.ArgumentParser()
parser.add_argument('-data_path', required=True, help="Base dir containing train, test, val directories of PCAM HDF5 Files")
parser.add_argument('-save_path', default='D:/PCAM DATA/trained_models', help="Path to save trained models")
parser.add_argument('-lr', default=1e-3, type=float, help="Learning rate")
parser.add_argument('-batch_size', default=32, type=int, help="Batch size")
parser.add_argument('-epochs', default=10000, type=int, help="Number of train iterations")
parser.add_argument('-device', default=0, type=int, help="CUDA device")
parser.add_argument('-save_freq', default=1000, type=int, help="Frequency to save trained models")
parser.add_argument('-visdom_freq', default=250, type=int, help="Frequency  plot training results")
args = parser.parse_args()
print(args)

# data
dataset_train = data_loader.PCamData(args.data_path, mode="train", batch_size=args.batch_size, augment=True)
dataset_valid = data_loader.PCamData(args.data_path, mode="validation", batch_size=args.batch_size)

device = 'cuda:{}'.format(args.device)

# Model
model = pretrained_network.PretrainedNet()
model.to(device)

# opt
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-8)

# loss
criterion = utils.loss

# Visdom writer
# writer = utils.Writer()

def train():
    model.train()
    
    losses = []
    for idx in range(1, args.epochs+1):
        #zero grad
        optimizer.zero_grad()
        
        # load data to GPU
        sample = dataset_train[idx]
        images = sample['images'].to(device)
        labels = sample['labels'].to(device)
        
        # forward
        predicted = model(images)
        
        # loss
        loss = criterion(predicted, labels)
        losses.append(loss.data.item())
        
        # back
        loss.backward()
        optimizer.step()
        
        print("Iteration: {:04d} of {:04d}\t\t Train Loss: {:.4f}".format(idx, args.epochs, np.mean(losses)), end="\r")
        
        # if idx % args.visdom_freq == 0:
        #     # get loss and metrics from validation set
        #     val_loss, accuracy, f1, specificity, precision = validation()
            
        #     # plot train and val loss
        #     writer.plot('loss', 'train', 'Loss', idx, np.mean(loss))
        #     writer.plot('loss', 'validation', 'Loss', idx, val_loss)
            
        #     # plot metrics
        #     writer.plot('accuracy', 'test', 'Accuracy', idx, accuracy)
        #     writer.plot('specificity', 'test', 'Specificity', idx, specificity)
        #     writer.plot('f1', 'test', 'F1', idx, f1)
        #     writer.plot('precision', 'test', 'Precision', idx, precision)
            
        #     # print output
        #     print("\nIteration: {:04d} of {:04d}\t\t Valid Loss: {:.4f}".format(idx, args.epochs, val_loss), end="\n\n")
            
        #     # set model to train
        #     model.train()
            
        if idx % args.save_freq == 0:
            torch.save(model.state_dict(), args.save_path+"/model-pretrained-{:05d}.pt".format(idx))
            
def validation():
    model.eval()
    
    losses=[]
    accuracy=[]
    f1=[]
    specificity = []
    precision = []
    for idx in range(len(dataset_valid)):
        sample = dataset_valid[idx]
        
        images = sample['images'].to(device)
        labels = sample['images'].to(device)
        
        predicted = model(images)
        
        loss = criterion(predicted, labels)
        losses.append(loss.data.item())
        
        metrics = utils.metrics(predicted, labels)
        accuracy.append(metrics['accuracy'])
        f1.append(metrics['f1'])
        specificity.append(metrics['specificity'])
        precision.append(metrics['precision'])
        
    return torch.tensor(losses).mean(), torch.tensor(accuracy).mean(), torch.tensor(f1).mean(), torch.tensor(specificity).mean(), torch.tensor(precision).mean()


if __name__ == "__main__":
    train()
        
            
            