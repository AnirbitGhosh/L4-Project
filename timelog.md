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
