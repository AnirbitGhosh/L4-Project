# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as plt
from PIL import Image, ImageDraw
import os
import torchvision.transforms as transforms
from torchvision import utils
import torch
from dataset import breastCancerDataset
from dataset import show

#%% 
# load data and set paths
data_base_path = "D:/PCAM DATA"
train_tiles_path = "D:/PCAM DATA/train/tiles"
train_meta_path = "D:/PCAM DATA/train/train_labels.csv"
train_data = pd.read_csv(train_meta_path)
train_data.head(5)

#%%
# count malignant vs benign 
counts_tumor_patch = train_data['tumor_patch'].value_counts()
counts_tumor_center_patch = train_data['center_tumor_patch'].value_counts()

print("Number of tumor tiles")
print(counts_tumor_patch)

print("Number of tiles with tumor in the center patch")
print(counts_tumor_center_patch)


# %%
# get malignant image ids
malignant_ids = train_data.loc[train_data['center_tumor_patch']== True ]['id'].values
print(malignant_ids[:10])

# %%
color = True

plt.rcParams['figure.figsize'] = (10,10)
plt.subplots_adjust(wspace=0, hspace=0)
nrows, ncols = 3,3

# %%
# plot images with rectangle around center 32x32 px
for i, id_ in enumerate(malignant_ids[:nrows*ncols]):
    print(id_)
    fname = os.path.join(train_tiles_path, str(id_) + '.jpeg')
    
    img = Image.open(fname)
    draw = ImageDraw.Draw(img)
    draw.rectangle(((32,32), (64,64)), outline='red', width=2)
    plt.subplot(3, 3, i+1)
    if color:
        plt.imshow(np.array(img))
    else:
        plt.imshow(np.array(img)[:,:,0], cmap='gray')
    plt.title(str(id_))
    plt.axis('off')
    
# %%
# load our custom dataset object
from dataset import breastCancerDataset

data_transformer = transforms.Compose([transforms.ToTensor()])
img_dataset = breastCancerDataset(data_base_path, data_transformer, "train")
print("Size of dataset: ", len(img_dataset), " images")

# %%
# load an image from dataset
img, label = img_dataset[256]
print(img.shape, torch.min(img), torch.max(img))

# %%
# display some images
grid_size = 4
rnd_inds = np.random.randint(0, len(img_dataset), grid_size)
print('image indices: ', rnd_inds)

# %%
x_grid_train = [img_dataset[i][0] for i in rnd_inds]
y_grid_train = [img_dataset[i][1] for i in rnd_inds]

x_grid_train = utils.make_grid(x_grid_train, nrow=4, padding=2)
print(x_grid_train.shape)

# %%
plt.rcParams['figure.figsize'] = (10.0, 5)
show(x_grid_train, y_grid_train, True)
