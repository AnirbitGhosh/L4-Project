#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))
print(sys.path)

#%%
# Import modules
import openslide
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from PIL import Image
import matplotlib.ticker as plticker
import copy

#%%
image_dir = "D:/PCAM DATA/WSI/Whole Slide Images"
tile_dir  = "D:/PCAM DATA/WSI/Tiles/TCGA-Slide-01"

print(os.listdir(image_dir))

# %%
# Opening image
slide = open_slide(image_dir + "/" + os.listdir(image_dir)[0])
slide_props = slide.properties

thumb = slide.get_thumbnail((600,600))
plt.figure(figsize=(10,10))
plt.imshow(thumb)
plt.axis("off")
# %%
print(f"The dimensions of the WSI in pixels is: {slide.dimensions}")

#%%
tiles = DeepZoomGenerator(slide, tile_size=int(96), overlap=0, limit_bounds=False)
level_num = tiles.level_count-1
cols, rows = tiles.level_tiles[level_num] # for whole slide's tiles
print(cols, rows)

#%%
df = pd.read_csv("D:/PCAM DATA/WSI/Tiles/TCGA-01-predictions.csv")
df['tile_coord'] = df['image'].str[:-4]
df.head()

#%%
thumb = np.array(slide.get_thumbnail((230, 329)))
print(thumb.shape)

# %%
plt.imshow(thumb)

# %%
masked_img = copy.deepcopy(thumb)
for index, row in df.iterrows():
    coord = row['tile_coord'].split('_')
    row_coord = int(coord[1])
    col_coord = int(coord[0])
    if row['predictions'] == 1:
        masked_img[row_coord][col_coord] = [255,0,0]
        # print(masked_img[row_coord][col_coord][0])
        # print(masked_img[row_coord][col_coord][1])
        # print(masked_img[row_coord][col_coord][2])
    else:
        masked_img[row_coord][col_coord] = [0,255,0]

plt.figure(figsize=(10,10))
plt.imshow(masked_img)
plt.axis("off")
# %%