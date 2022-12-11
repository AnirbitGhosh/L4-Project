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

## Week 6 

## 05 Nov 2022
* *1 hour* : Complete predicition generator to produce binary output given a directory containing whole slide images, network weight (.pt file) and an output directory. It creates a CSV file, containing tile coordinates against predicted class. Generalize it for eaiser portability across computers.
* *2 hours* : Run prediction pipeline on all whole slide images and generate all class labels. Run the labels through the tile visualizer to see prediction visualized. RESULTS: Most of the slide is viosualized as positive with edges being negative, unsure about its accuracy and validity since TCGA data is not annotated

## 07 Nov 2022
* *1 hour* : Study CBioPortal clinical data for survival times. Download KM plot data to get a dataset of all patients for whom surival time and deceased status is available. 
* *1 hour* : Test validity of prediction network by extracting tissue slide for 1. high survival time patient 2. low survival time patient. Use TCGA Patient ID from downloaded KM plot data,  to access patitent slides on TCGA.

## 08 Nov 2022
* *1 hour* : Train network on 10k PCAM tiles to improve accuracy od predictions. Decide to use own custom network as ResNet is taking too long to train
* *1 hour* : Tile newly downloaded WSI for two patients corresponding to low and high survival times. Use tiles to generate predictions from the trained network. 
* *1 hour* : analyze predictions and visualization produced from predictions. Predictions tend to no longer be biased towards just positive. More negatives produced. Still negatives focussed around edges and interior is all positive. 
* *0.5 hour* : Calculate average malignancy score by taking the percentage of tiles/pixels that are classified as positive from the total number of tiles. Rationale: Gives a percentage to quantify severity of tumour. calculated score gave 80% positive for higher survival time tissue slide and 60% for lower survival time, which is inverse of what was expected. FACTORS TO CONSIDER: AGE was different - 1. lower survival time sample had a much older patient vs much younger patient for higher survival time. 2. Image size was different, making them have different number of total tiles/pixels, causing percentage to not be a good measure. 

## 09 Nov 2022
* *2 hours* : Supervisor meeting - noted in minutes

## Week 7

## 12 Nov 2022
* *2 hours* : Try to train resnet again with less data, turns out to be have many more complications. Current image training data is 96x96 while RESNET requires 244x244. I resized images using tensor transformations to match required dimensions, but artificially increasing dimensions adds to noise. Second complication is that training with less data to try and get quicker computation leads to very low accuracy, almost 70% (<75% for custom network). 
* *1 hour* : Run prediction pipeline using pretrained network, generate full class labels for all 3 WSI images. Visualize predictions as images. Produces much worse predicitions, with almost entire slide being predicted as positive, with almost less than 5% tiles being predicted negative. Clear bias towards positive class, indicative of underfitting due to insufficient data. Requires much more data to obtain a good model with Resnet.

## 14 Nov 2022
* *1 hour* : Read https://medium.com/the-researchers-guide/survival-analysis-in-python-km-estimate-cox-ph-and-aft-model-5533843c5d5d
* *1 hour* : Read https://www.kdnuggets.com/2020/07/guide-survival-analysis-python-part-3.html

## 15 Nov 2022
* *1 hour* : Investigate ways to extract features from image data to use in cox hazard model to develop survival model. Initially use about 20 WSI images for feature extraction to see if Cox Model actually gives reasonable results before expanding to more data. 
* *1 hour* :  Read https://www.nature.com/articles/s41598-022-19112-9 - image based survival prediction, but uses a grading system developed with a multi-instance learning model which is out of scope for this project and very different from what we are trying to achieve

## 16 Nov 2022
* *3 hours* : Supervisor Meeting - noted in minutes

## 18 Nov 2022
* *1 hour* : Read https://academic.oup.com/bioinformatics/article/38/14/3629/6604265?rss=1 - breast cancer image survival analysis
  
## Week 8 

NO WORK DONE DUE TO MAJOR DEADLINES AND 5 COURSEWORKS TO COMPLETE

## Week 9

## 28 Nov 2022
* *1.5 hours* : Extract 100k training and test tiles from PCAM data, creating full final dataset for training network
* *1 hour* : Train custom neural net with full 100k dataset - Noticed accuracy did not improve too much despite a 10x increase in training data.

## 29 Nov 2022
* *1 hour* : Fine tune network parameters, plotting accuracy and loss for different parameters to see what works best. Using TanH over ReLU showed highest accuracy although the training process showed a lot of fluctuation with TanH compared to ReLU. With ReLU accuracy remained more stable during training. But TanH gave 90% accuracy vs ReLU gave 75%. Train final network with 100k, full data set and complete training stage of model.
* *1.5 hour* : Run prediction pipeline on 3 existing WSI images, 1 control image, 1 high survival image, 1 low survival image. Visualize predicted classes as image mask and analyse tumour severity from malignancy score. Control image was mostly negative now, with a small concentration of positive tiles indicating a more realistic tumour. High vs low survival images also showed predictions more resemblant to their survival times. 
* *1 hour* : Download 20 WSI from TCGA corresponding to Patient IDs from CBioPortal survival time data. A lot of patient images could not be downloaded as the slides are not suitable for this task. They have different dye stains, some have poor quality, with mostly blank, some are multi tissue in a single slide (cant use that, it will ruin consistency if some images have more tiles due to having multiple tissue sections, aim to get all tisues which are a single tissue piece), out of 45 first patients only 20 were usable. Mark unusable ones as 0, mark questionable ones as 2 or 3 and discuss with supervisor

## 30 Nov 2022
* *2 hours* : Supervisor meeting - noted in minutes

## 1 Dec 2022
* *1 hour* : Fix issues with tiling code and re run tiling-prediction pipeline on linux machine 

## 2 Dec 2022
* *1 hour* : Do reserach on BRCA annotated data. Download QUPATH, try to load annotations onto WSI from https://zenodo.org/record/5320076#.Y4olBTPP0UE, unsure how to load CSV annotations (in x and y coord format) into QUPATH. Check CAM17 database to find annotated WSI -  downloading the WSI showed extremeely slow download seeds, a 1gb image taking 30 mins to download - could not proceed with validating PCAM data.
* *1 hour* : Read macenko normalization paper (https://www.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf), and look into implementing some form of h&e normalization step to pre-process tiles before being used to train network and predict
* *2 hours* : Implement Macenko normalization, H and E channel extraction code. Train network with Normalized images
* * 4 hours* : Run predictions using macenko normalized network on 1. Un-normalized WSI tiles and 2. Normalized WSI tiles - Results seem worse when normalized network is used on normalzied WSI tiles. However, using un-normalized network on normalized tiles produced reasonable predictions. Using normalized network on un-normalized tiles produces biased predictions towards +ve tiles. Consider optimizing network to be able to use normalized WSI with non-normalized network? so far predictions without normalization seem more resonable maybe because PCAM data is already normalized? 

## Week 10

## 03 Dec 2022
- *2 hours* : Found CAMELYON 17 data and annotations. Opened annotated slide on ASAP 2.1. Tumour seems to be a very small region of the whole slide, rest is non-malignant tissue. Test what our network predicts on this CAMELYON 17 slide. 
- *2 hour* : Run camelyon WSI, a test slide and a normal tissue sample through our models - 2 networks : 1. trained with default PCAM tiles 2. trained with Macenko PCAM tiles. Then, Macenko normalize and tile every WSI from CAMELYON 17 repo and run predictions on them. Results show more accurate predictions to the ground truth annotation for non-normalized network + macenko WSI tiles. For normalized network + macenko WSI, there is a lot of overprediction of +ve tiles likely because PCAM data is infact already normalized to maintain consistency and applying macenko again ruins data quality. 
- *3 hours* : From https://portal.gdc.cancer.gov/repository - Manually inspect and download each deceased patient's (who has a survival time data available) biopsy slide from TCGA, ensuring only appropriate slide - Single tissue, decent tissue volumne (not mostly empty space) and not significant debris. Total WSI downloaded = 54 in this set. Previous set had 20 WSI. TOTAL FINAL DATASET HAS 74 WSI, with corresponding survival data to use for modeling survival time. 

## 04 Dec 2022
- *2 hours* : Refine tumor prediction data by calculating malignancy score - quantified as the % of positively predicted tiles from the whole image. Use Lifelines library to implemnt KM plot on survival data from CBioPortal
- *2 hours* : Use Lifelines model to implement Cox Hazard model on the tumor data and the survival data from CBioPortal. Use Tumour data of the first 20 WSI predictions, matching it to the corresponding survival data with patient ID. Use OS_MONTHS, OS_STATUS and Malignancy Score as the covariate to model the survival of patients correlated with the tumour severity scores predicted form Deep Learning pipeline. Plot CPH estimate graph for any arbitrary malignancy score and compare to baseline fit model. 
- *2 hours* : Perform some baseline statistical analysis on CPH model fitted using Malignancy score to see statistical significance. Use proportional_hazard_test() from Lifelines library to generate a p value of the correlation between survival time and malignancy score, giving p=0.7

## 05 Dec 2022
- *2 hours* : Noticed PCAM data is undersampled to 10x magnification. Modify code to use 4x downsampling level to allow WSI to be sampled at 40/4 = 10x magnification. Re run prediction pipeline on newly downsapled WSI tiles. 
- *1 hour* : Get more CAM17 data, a normal and a tumour slide. Run it through model and compare predictions to annotations. Reveals very accurate predictions, normal slides being predicted as completely negative and positive slides showing slight overprediction but the positive region is so small it is impossible to get accurate predictions. 
- *1 hour* : Compare predicition quality using 1. normalized model on downsampled tiles vs 2. un-normalized model on downsampled tiles. This showed using un-normalized model had the better prediction than the normalized on CAM17 data, closer to expected annotation. So it is used as a validation benchmark to use for TCGA WSI slides, with 10x undersampling and macenko normalization on WSI tiles and DCNN trained on non-normalized model. 

## 06 Dec 2022
- *1 hour* : Prepare code to be run on Linux workstation to process full WSI dataset of 74 WSIs and generate predcitions for all images. 
- *1 hour* : Transfer all WSI to Linux machine, write a script to automate running all piepline commands, tile-slide, generate-predictions and visualize-tiles
- *5 hours* : Run full dataset, 74gb of WSI through prediction pipleine to generate output.

## 07 Dec 2022
- *2 hours* : Supervisor meeting - noted in minutes
- *4 hours* : Re run full dataset of 74 WSIs through prediction pipeline to generate predictions and visualize predicted tumours - as the pipeline failed after tiling stage due to a line of code not having been changed for the Linux workstation

## 08 Dec 2022
- *2 hours* : Supervisor meeting (contd). Collect all data generated from prediction pipeline. Try transfer all tiles to HDD but ran out of space. Agree to not transfer tiled WSIs, instead work on Linux workstation by SSH-ing in over the holidays. 
- *1 hour* : Setup SSH connection for my personal user account on Linux workstation - Create user account, transfer all relevant files to my account, set ownership, create conda environment with all relevant packages. 
- NO OTHER WORK DONE AS EXAM ON 09-12

## 09 Dec 2022
- EXAM - NO WORK DONE

## 10 Dec 2022 
- BREAK

## Week 11

## 11 Dec 2022