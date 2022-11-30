from openslide import open_slide
import openslide
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import sys
import os
from pixel_analysis import find_mean_std_pixel_value
from openslide.deepzoom import DeepZoomGenerator
import tifffile as tiff



if __name__ == '__main__':
    for f in os.listdir(sys.argv[1]):
        file_path = os.path.join(sys.argv[1], f)
        slide = open_slide(file_path)
        # Load the slide file (svs) into an object
        slide_props = slide.properties
        print(slide_props)

        print("Vendor is:", slide_props["openslide.vendor"])
        print("Pixel size of X in um is:", slide_props["openslide.mpp-x"])
        print("Pixel size of Y in um is:", slide_props['openslide.mpp-y'])

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

        #################################

        # Generate object for tiles using the DeepZoomGenerator
        tiles = DeepZoomGenerator(slide, tile_size=int(sys.argv[3]), overlap=0, limit_bounds=False)
        #Here, we have divided our SVS file into tiles of size 244x244 with no overlap

        #The tiles object also contains data at many levels. 
        #To check the number of levels
        print("The number of levels in the tiles object are: ", tiles.level_count)
        print("The dimensions of data in each level are: ", tiles.level_dimensions)
        #Total number of tiles in the tiles object
        print("Total number of tiles = : ", tiles.tile_count)
        
        # How many tiles at a specific level
        level_num = tiles.level_count-1
        print("Tiles shape at level ", level_num, " is: ", tiles.level_tiles[level_num])
        print("This means there are ", tiles.level_tiles[level_num][0]*tiles.level_tiles[level_num][1], " total tiles in this level")

        

        ### Saving each tile to local directory
        cols, rows = tiles.level_tiles[level_num] # for whole slide's tiles
        #cols, rows = 100, 100

        tile_dir = sys.argv[2]
        sub_dir = os.path.basename(file_path)
        sub_dir_path = os.path.join(tile_dir, sub_dir)
        sub_dir_path = os.path.splitext(sub_dir_path)[0]
        os.mkdir(sub_dir_path)

        for row in range(rows):
            for col in range(cols):
                tile_name = str(col) + "_" + str(row)
                #print("Now getting tile with title: ", tile_name)
                temp_tile = tiles.get_tile(level_num, (col, row))
                temp_tile_RGB = temp_tile.convert('RGB')
                temp_tile_RGB_np = np.array(temp_tile_RGB)

                # only save tiles with >50% tissue samples
                # remove blank and partial tiles            
                if temp_tile_RGB_np.std() > 15 and temp_tile_RGB_np.mean() < 230: 
                     print("saving tile number:", tile_name)
                     tiff.imsave(sub_dir_path + "/" +tile_name + ".tif", temp_tile_RGB_np)
    


