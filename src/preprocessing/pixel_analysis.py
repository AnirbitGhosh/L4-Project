import matplotlib.pyplot as plt
import tifffile as tiff
import numpy as np
import glob
import sys


#Let us define a function to detect blank tiles and tiles with very minimal information
#This function can be used to identify these tiles so we can make a decision on what to do with them. 
#Here, the function calculates mean and std dev of pixel values in a tile. 
def find_mean_std_pixel_value(img_list):
    avg_pixel_value =[]
    stddev_pixel_value = []
    for file in img_list:
        image = plt.imread(file)
        avg = image.mean()
        std = image.std()
        avg_pixel_value.append(avg)
        stddev_pixel_value.append(std)
        
    avg_pixel_value = np.array(avg_pixel_value)
    stddev_pixel_value = np.array(stddev_pixel_value)
    
    print("Average pixel value for images is: ", avg_pixel_value.mean())
    print("Average std dev of pixel values for images is: ", stddev_pixel_value.mean())
    
    return (avg_pixel_value, stddev_pixel_value)

if __name__ == '__main__':
    orig_tile_dir = sys.argv[1]
    img_list = glob.glob(orig_tile_dir + "*.tif")
    
    img_stats = find_mean_std_pixel_value(img_list)