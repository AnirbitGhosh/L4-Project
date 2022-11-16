# Timelog

* Prediction of overall survival time from breast cancer H&E whole slide biopsy images 
* Anirbit Ghosh
* 2439281g
* Kevin Bryson


## Week 1

### 28 Sept 2022

* Project allocation released
* *0.5 hour* : read up on project description provided with proposal
* *0.5 hour* : contact supervisor to setup meeting for 30 Oct 

## 30 Sept 2022

* *2 hour* : meeting with supervisor

## 01 Oct 2022

* *2 hour* : "read Deep neural network models for computational histopathology: A Survey" provided by supervisor
* *1 hour* : setup tools required for initial stages of project

## 02 Oct 2022

* *1 hour* : annotate research paper and prepare notes for next supervisor meeting 
* *1 hour* : Research - find relevant journals and academic papers and add to ref manager
* *0.5 hour* : go through Deep Learning MSc Moodle page

## 03 Oct 2022

* *1 hour* : Read Week 1 & 2 Deep Learning material from DL MSc course
* *2 hour* : Finish pytorch blitz tutorial (Week 1 lab DL Msc)
* *0.5 hour* : Create Github Repo and setup repo from template

## 04 Oct 2022

* *2.5 hour* : Supervisor Meeting - noted minutes

## Week 2 

## 06 Oct 2022

* *1 hour* : Open-Slide python tutorial - Learn basics of tiling whole slide images
* *2 hours* : Read and annotate "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study"

## 07 Oct 2022

* *1 hour* : Read and annotate "Predicting survival from colorectal cancer histology slides using deep learning: A retrospective multicenter study"
* *2 hours* : create pipeline with open-slide for tiling slides at native resolution

## 09 Oct 2022

* *1 hour* : Research ways to filter background tiles from tissue tiles
* *1 hour* : Read documentation for wsi-tile-cleanup python library to use for optimizing tile creation and remove empty tiles

## 10 Oct 2022

* *0.5 hour* : research ways to deal with WSI storage issue - possibly store on university Network Drive
* *0.5 hour* : plan working methodolgy steps - training -> tissue decomposition -> tumour grading -> survival time

## 11 Oct 2022

* *1 hour* : https://www.youtube.com/c/DigitalSreeni/videos - videos on WSI H&E filtering and normalization
* *1 hour* : Perform pixel level analysis to detect blank, partial and goot WSI tiles
* *1 hour* : Modify existing tiling pipeline to accept command line arguments and include pixel data to filter useless tiles
* * 2 hours* : Supervisor meeting - noted minutes


## Week 3

## 14 Oct 2022

* *0.5 hour* : Watch histology and segmentation video on Camelyon17 challenge : https://www.youtube.com/watch?v=8h6oSqPrjzc 
* *1 hour* : Watch video on brain cancer detection using TensorFlow MobileNet pretrained transfer learning :  https://www.youtube.com/watch?v=7MceDfpnP8k&list=PLoOXIgpPlgrg3bWuK9rRDj29nlx7Mt6KG&index=2&t=4104s  
* *0.5 hour* : Understand existing literature of Camelyon challenge solution on Breast cancer metastasis detection: https://github.com/alexmagsam/metastasis-detection 

## 16 Oct 2022
* *1 hour* : Read Camelyon16 challenge documentation and study PCAM data : https://github.com/basveeling/pcam, https://camelyon16.grand-challenge.org/ 
* *1 hour* : Download PCAM dataset and write data loader code to convert HDF5 format to image format : extract-data.py
* *1 hour* : Read article on "Histopathological Cancer Detection with Deep Neural Networks" using PCAM dataset : https://humanunsupervised.github.io/humanunsupervised.com/pcam/pcam-cancer-detection.html
* *1 hour* : Watch a Kaggle histopathology challenge code review on cancer tissue segmentation : https://www.youtube.com/watch?v=qxtDv7_U0hY&list=PLoOXIgpPlgrg3bWuK9rRDj29nlx7Mt6KG&index=7&t=339s
* *1 hour* : Read differences between CNN using TorchVision or TensorFlow (Keras), start initial coding of CNN to process acquried data.

## 18 Oct 2022
* *1 hour* : Create custom dataset class inheriting torch.data.Dataset to hold images + labels in one datastructure
* *1 hour* : use VSCode interactive mode to visualize images inline, instead of using online notebook
* *0.5 hour* : Load training data into custom data model, visualize random images in grids and highlight malignant areas in tumour annotated images
* *2 hours* : Supervisor Meeting (noted in minutes)

## Week 4 

## 21 Oct 2022
* *1 hour* : Read Pytorch transfer learning documentation - https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
* *0.5 hour* : Read up on different pre trained networks, Resnet, mobilenet, VGG16/19

## 22 Oct 2022
* *4 hours* : Create custom CNN model, train on PCAM data and validate on PCAM data - accuracy 73% achieved. 

## 24 Oct 2022
* *2 hours* : Create a predictor program to load the PCAM data trained model weights to predict malignancy of TCGA WSI image tiles - plot images with color coded to indicate malignancy (red) and benign (gray).

## 25 Oct 2022
* *2.5 hours* : Supervisor meeting - noted in minutes

## 27 Oct 2022
* *2 hours* : Create a Resnet18 model using Pytorch docs, use all intermediate layers and FC layers at first. Split image dataset into malignant and benign directories for pytorch ImageLoaders to work and extract classes automatically inside the network
* *1 hour* : Try debug why Resnet18 wont train/training is extremely slow : use smaller dataset, which smaller batch number - not change. use less epochs to train, still very slow. 

## Week 5

## 30 Oct 2022
* *2 hours* : Refine Resnet18 model by freezing layers to increase speed. Train my own model with more data and random transforms to increase robustness - 10k data used and highest validation acc obtained of 76%. Save model weights and use for prediction on TCGA slide
* *0.5 hour* : Tile an entire TCGA whole slide image into tiles, discarding background. 
* *1 hour* : Create a prediction pipeline to automatically iterate through every tile given a directory, load our model, pass each tile as input and generate a prediciton. Save binary prediction in a CSV file against the file name. 
* *1 hour* : use generated predictions to conditionally apply a mask to tiles. Visualize malignant tiles vs benign tiles, notice a lot more malignant compared to benign, very noise when used on out of sample data. Try ways of combining tiles back into a full image to see the overall areas of malignancy and quantify noise generated. 

## 02 Nov 2022
* *2.5 hours* : Supervisor Meeting - noted in minutes

## 03 Nov 2022
* *4 hours* : Investigate multiple ways of generating overall visualization of tiles : 
    * Concatenate each tile as a numpy array into another array and try visualize that. Shaped were not compatible and collapsing additional dimensions generated arrays too big to store in memory as we have a lot of tiles
    * Try draw rectangles of size 96x96 on regions where our tiles were extracted from our whole slide. However, drawing rectangles requires the coordinate of the center pixel and tile coordinates are not coordinates of their central pixel, they are the number of the tile itself in a row-col grid laid over the whole image. This approach failed. 
    * Use Matplotlib grids to replicate similar grid on top of whole slide image preview and colour code each grid square corresponding to tile coordinate based on predicted values stored in CSV - Problem is displaying the whole slide image in native resolution is not possible and as soon as we downsample it distorts the tile coordinates and they no longer apply
    * SOLUTION : Scale whole slide image down by x96 to represent each tile as a pixel on the image. Now we can modify each pixel's colour based on the corresponding tile's predictions. This worked to generate a cummulative visualization of our predictions

## 04 Nov 2022
* *2 hours* : Create a tile visualizer script to take tile level predictions and map it to apply a pixel level mask on top of whole slide image thumbnail preview image scaled down by a factor of 96. Save the masked image as a png in image directory. 
* *1 hour* : Create generic prediction generator to produce class outputs for all whole slide image tiles given a model weight .pt file and saves output in -o directory