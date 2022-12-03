#%%
# Import modules
import os
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import argparse
from PIL import Image
import copy
import sys

#%%
def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
#%%
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--image', type=dir_path, help='Pass directory containing whole slide images with -i or --image flags', default="D:/PCAM DATA/WSI/Whole Slide Image")
parser.add_argument('-l', '--label', type=dir_path, help="Pass directory containing CSV files with predicted lables using -l or --label flags")

#%%
def process_labels(image_name, label_dir):
    fname = image_name[:-4] + "-predictions-normalized.csv"
    fname = os.path.join(label_dir, fname)
    df = pd.read_csv(fname)
    df['tile_coord'] = df['image'].str[:-4]
    
    predictions = df['predictions'].values.tolist()
    images = df['tile_coord'].values.tolist()
    
    data = {}
    for key in images:
        for value in predictions:
            data[key] = value
            predictions.remove(value)
            break
    
    return data

def process_image(image_name, image_dir):
    slide = open_slide(image_dir + "/" + image_name)
    tiles = DeepZoomGenerator(slide, tile_size=int(96), overlap=0, limit_bounds=False)
    
    level_num = tiles.level_count-1
    cols, rows = tiles.level_tiles[level_num]
    
    thumb = slide.get_thumbnail((cols, rows))
    #plt.figure(figsize=(10,10))
    #plt.imshow(thumb)
    #plt.axis("off")
    
    return thumb

def generate_pixel_map(thumb, predictions):
    np_thumb = np.array(thumb)
    for coord, pred in predictions.items():
        split_coord = coord.split('_')
        col_coord, row_coord = int(split_coord[0]), int(split_coord[1])
        
        if pred == 1:
            np_thumb[row_coord][col_coord] = [255, 0, 0]
        else:
            np_thumb[row_coord][col_coord] = [0, 255, 0]
    
    return np_thumb
                
def plot_images(image_list):
    row = len(image_list)//2
    col = len(image_list)//2
    
    if row == 0:
        row = 1
        col = 1
    
    for num, img in enumerate(image_list):
        plt.subplot(row, col, num+1)
        plt.axis("off")
        plt.imshow(img)
        plt.show()

def save_masked_image(img_arr, image, output_dir):
    fname = os.path.join(output_dir, image[:-4]+"-myCustom.png")
    plt.imsave(fname, img_arr, format="png")

#%%
if __name__ == "__main__":
    if len(sys.argv) < 4:
        parser.print_usage()
        parser.print_help()
        sys.exit(1)
        
    args = parser.parse_args()
    
    print("Processing arguments...")
    img_dir = args.image
    label_dir = args.label
    
    print(f"Whole Slide Image directory : {img_dir}")
    print(f"Predicted Labels directory : {label_dir}")
    
    print("iterating over images in image directory...")
    
    image_list = []
    for file in os.listdir(img_dir):
        prediction_dict = process_labels(file, label_dir)
        
        thumbnail = process_image(file, img_dir)

        np_img_arr = generate_pixel_map(thumbnail, prediction_dict)
        # image_list.append(np_img_arr)
        
        save_masked_image(img_arr=np_img_arr, image=file, output_dir=label_dir)
        
    #plot_images(image_list)
    
