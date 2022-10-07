from openslide import open_slide
import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# Load the slide file (svs) into an object
# slide = open_slide("WSI-data/TCGA-AO-A126-01Z-00-DX1.D9D6AA15-32F0-44BD-AF30-36191784FFA2.svs")
slide = open_slide("WSI-data/sample_WSI_TCGA.svs")

slide_props = slide.properties
print(slide_props)

print("Vendor is:", slide_props["openslide.vendor"])
print("Pixel size of X in um is:", slide_props["openslide.mpp-x"])
print("Pixel size of X in um is:", slide_props['openslide.mpp-y'])

# Objective used to capture the image
objective = float(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
print("The objective power is:", objective)

# Get slide dimensions for the level 0 - max resolution level
slide_dims = slide.dimensions
print("Dimensions of the slide at level 0 - max res: ", slide_dims)

# Get a thumbnail of the image and visualize
slide_thumb_600 = slide.get_thumbnail(size=(600,600))
# slide_thumb_600.show()

# convert thumbnail to np array
slide_thumb_600_np = np.array(slide_thumb_600)
plt.figure(figsize=(8,8))
plt.imshow(slide_thumb_600_np)
# plt.show()

# get slide dims at each level. Remeber that a whole slide image stores information as pyramid at various levels
dims = slide.level_dimensions
num_level = len(dims)
print("Number of levels in the iamge are:", num_level)
print("Dimensions of various levels in this image are:", dims)

# By how much are levels downsampled from the original image?
factors = slide.level_downsamples
print("each level is downsampled by an amount of:", factors)

# Copy an image from a level 
level3_dim = dims[2]
# Give pixel coordinates (top left pixel in the original large image)
# Also give the level number (for level 3 we are providing a valueof 2)
# Remember that the output would be a RGBA image (NOT RGB)
level3_img = slide.read_region((0,0), 2, level3_dim) # Pillow object, mode=RGBA

# convert the iamge to RGB
level3_img_RGB = level3_img.convert('RGB')
#level3_img_RGB.show()

# Return the best level for displaying the given downsample
SCALE_FACTOR =  52
best_level = slide.get_best_level_for_downsample(SCALE_FACTOR)
print("best level: ", best_level)
#Here it returns the best level to be 2 (third level)
#If you change the scale factor to 2, it will suggest the best level to be 0 (our 1st level)
#################################

#Generating tiles for deep learning training or other processing purposes
#We can use read_region function and slide over the large image to extract tiles
#but an easier approach would be to use DeepZoom based generator.
# https://openslide.org/api/python/

from openslide.deepzoom import DeepZoomGenerator

# Generate object for tiles using the DeepZoomGenerator
tiles = DeepZoomGenerator(slide, tile_size=256, overlap=0, limit_bounds=False)
#Here, we have divided our SVS file into tiles of size 244x244 with no overlap

# Tiles contain meta data at many levels
# To check numebr of levels
print("The number of levels in the tiles object are:", tiles.level_count)

print("The dimensions of data in each level are:", tiles.level_dimensions)

# the number of tiles in the tiles object
print("Total number of tiles: ", tiles.tile_count)

# How many tiles at a specific level
level_num = 16
print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])
print("This means there are ", tiles.level_tiles[level_num][0]*tiles.level_tiles[level_num][1], " total tiles in this level")

# Dimensions of the tile (tile size) for a specific tile from a specific layer
tile_coord = (7,4)
tile_dims = tiles.get_tile_dimensions(11, tile_coord)
print("At level 11, the size of tile ", tile_coord, " is: ", tile_dims)

# tile count at the highest resolution level (level 17) in our tiles
tile_count_in_large_image = tiles.level_tiles[16]
print("tile count at the highest resolution level (level 17) in our tiles: ", tile_count_in_large_image)

# Check tile size for some random tile
tile_dims = tiles.get_tile_dimensions(16, (120,140))
print("Tile dimensions for some random tile at max resolution level: ", tile_dims)

# Check tile size for last tile it may not have full 244x244 dimensions
tile_dims = tiles.get_tile_dimensions(16, (225, 150))
print("tile size for last tile at level 16: ", tile_dims )

single_tile = tiles.get_tile(16, (62, 72)) # provide deep zoom level and address
single_tile_RGB = single_tile.convert('RGB')
#single_tile.show(single_tile)

#### Saving each tile to local directory
# cols, rows = tiles.level_tiles[16] # for whole slide's tiles
cols, rows = 75, 120
import os
tile_dir = "WSI-data/sample_WSI_TCGA_tiles/"

for row in range(115, rows):
    for col in range(70, cols):
        tile_name = os.path.join(tile_dir, '%d_%d' % (col, row))
        print("Now saving tile with title: ", tile_name)
        temp_tile = tiles.get_tile(16, (col, row))
        temp_tile_RGB = temp_tile.convert('RGB')
        temp_tile_RGB_np = np.array(temp_tile_RGB)
        plt.imsave(tile_name + ".png", temp_tile_RGB_np)


