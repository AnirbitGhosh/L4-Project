import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
import os
import tifffile as tiff

"""
LOAD PCAM Data
This script reads in a base directory containing the PCAM HDF5 datasets and iterates through the training, test and validation datasets to save
the images into their own directories

__/basedir
          |__ /train
                |__ camelyonpatch_level_2_split_train_x.h5
                |__ train_labels.csv
                |__ /tiles
          |__ /test
                |__ camelyonpatch_level_2_split_test_x.h5
                |__ test_labels.csv
                |__ /tiles
          |__ /validation
                |__ camelyonpatch_level_2_split_valid_x.h5
                |__ validation_labels.csv
                |__ /tiles

The tiles are stored within the /tiles sub-directory inside each of the three dataset directories. 
We use a subset of the entire dataset for training, testing and validation. 
Using a 1000 tiles for training, 100 for testing and validation.
Each directory also stores the metadata as a CSV file containing the labels of each tile corresponding to their filename [0,1,2....n]


- [1] B. S. Veeling, J. Linmans, J. Winkens, T. Cohen, M. Welling. "Rotation Equivariant CNNs for Digital Pathology". arXiv [cs.CV] (2018), (available at http://arxiv.org/abs/1806.03962).
Source: https://github.com/basveeling/pcam
"""


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=dir_path, help='Pass base directory path name with -p or --path flags')

def extract_data(basedir):
    
    print("Setting up directories")
    train_dir = basedir + "/train"
    test_dir = basedir + "/test"
    valid_dir = basedir + "/validation"
    
    print("Extracting training images...")
    
    #extract training images as .jpeg files
    with h5py.File(train_dir+"/camelyonpatch_level_2_split_train_x.h5", 'r') as h5f:
        for i in range(h5f['x'].shape[0])[:100000]:
            print("Saving image: " + str(i))
            img_arr = h5f['x'][i]
            plt.imsave(train_dir + "/tiles/" + str(i) + ".jpeg", img_arr)

    #extract test images as .jpeg files
    # with h5py.File(test_dir+"/camelyonpatch_level_2_split_test_x.h5", 'r') as h5f:
    #     for i in range(h5f['x'].shape[0])[:20000]:
    #         print("Saving image: " + str(i))
    #         img_arr = h5f['x'][i]
    #         plt.imsave(test_dir + "/tiles/" + str(i) + ".jpeg", img_arr)
     
    # print("Extracting Validation images...")
    # #extract validation images as .jpeg files       
    # with h5py.File(valid_dir+"/camelyonpatch_level_2_split_valid_x.h5", 'r') as h5f:
    #     for i in range(h5f['x'].shape[0])[:500]:
    #         print("Saving image: " + str(i))
    #         img_arr = h5f['x'][i]
    #         plt.imsave(valid_dir + "/grouped_tiles/"+ str(i) + ".jpeg", img_arr)
    

if __name__ == "__main__":
    args = parser.parse_args()
    print("Parsing arguments...")
    basedir = args.path
    
    print("Base directory received as: " + basedir)
    print("Starting data extraction...")
    extract_data(basedir)

    print("Successfully completed")