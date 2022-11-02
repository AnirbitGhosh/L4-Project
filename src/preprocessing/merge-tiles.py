#%%
from openslide import open_slide
import openslide
from openslide.deepzoom import DeepZoomGenerator
from matplotlib import pyplot as plt
import pyvips
import os
import sys

#%%
print(os.environ['PATH'])

#%%
base_path = "D:/PCAM DATA/WSI/"

print("number of tiles in directory: " + len(os.listdir(os.path.join(base_path, "Tiles/TCGA-slide-01"))))


#%%
file_path = "D:\PCAM DATA\WSI\Whole Slide Images\TCGA-Slide-01.svs"
slide = open_slide(file_path)

tiles = DeepZoomGenerator(slide, tile_size=96, overlap=0, limit_bounds=False)
level_num = tiles.level_count-1
col, row = tiles.level_tiles[level_num]

tile_arr = [pyvips.Image.new_from_file(f"{x}_{y}.tif", access="sequential") for y in range(row) for x in range(col)]
image_merged = pyvips.Image.arrayjoin(tile_arr, across=col)
plt.imshow(image_merged)
