# Timelog

* Prediction of overall survival time from breast cancer H&E whole slide biopsy images 
* Anirbit Ghosh
* 2439281g
* Kevin Bryson


## Week 1 - 14.5 hours

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

## Week 2 - 14 hours

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
* *2 hours* : Supervisor meeting - noted minutes


## Week 3 - 11.5 hours

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

## Week 4 - 13 hours

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

## Week 5 - 14 hours

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

## Week 6 - 10.5 hours

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

## Week 7 - 11 hours

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

## Week 9 - 17 hours

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
* *4 hours* : Run predictions using macenko normalized network on 1. Un-normalized WSI tiles and 2. Normalized WSI tiles - Results seem worse when normalized network is used on normalzied WSI tiles. However, using un-normalized network on normalized tiles produced reasonable predictions. Using normalized network on un-normalized tiles produces biased predictions towards +ve tiles. Consider optimizing network to be able to use normalized WSI with non-normalized network? so far predictions without normalization seem more resonable maybe because PCAM data is already normalized? 

## Week 10 - 35 hours

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

## Week 11 - 25 hours

## 11 Dec 2022
- *3 hours* : Implement probability predicting model instead of binary class prediction. Binary class prediction thresholds at 0.5 so whichever class is higher than 0.5 probability gets outputted as the class. This overpredicts malignant tiles by a lot even where tiles have a low probability of being positive, almost close to 0.5 but slightly higher than negative. Using probability of malignancy as the ouput rather than discrete class predictions allows to plot heatmaps of malignancy showing areas of highest probability of malignancy which is a lot more accurate than approximating whole tiles as 0 or 1. Using probability predictions we can now set our own threshold at 0.7 or higher to pick regions of the tissue with highest probability of being malignant to represent the true tumour region, and use that for malignancy score calculation. This will reduce noise in the calculated score, by not overpredicting malignant tiles and only using the tiles with the highest confidence. 
- *2 hours* : implement new heatmap visualization pipeline to use probability values rather than discrete class predictions to display the masked tissue thumbnail. Take the thumbnail of WSI at the level tiles were sampled, then convert it to greyscale 96x96x1 image with 1 colour channel. Then normalize the image by dividing pixels by 255. This normalized, greyscaled thumbnail will be masked with the prediction values. Using tile coordinates set the value of each pixel on the thumbnail to its corresponding probability value between 0 and 1. Since the rest of the image was also normalized by 255, all pixel values are between 0 and 1 and will allow us to use a continuous colour map ranging between 0 and 1. Finally plot the thumbnail with masked probability values inserted using a continuous colour map, I used 'hot_r' to represent the heatmap. Then plot the colour bar as well to show the meaning of various colors. Run into a small issue of visibility in the plotted image, as the image was very small compared to the massive color bar when the images were horizontally orientated and the colourbar was vertical. Using scipy.ndimage.rotate() to rotate the probability masked thumbnail to make it vertical and orient along the colourbar, with a dpi=200 to show a heatmap with more clarity and detail.

## 13 Dec 2022
- *3 hours* : Run full prediction pipeline for probability predictions on all 74 WSI tiles on Linux machine using SSH
- *0.5 hour* : Run heatmap visualization on all predictions generated to create heatmaps for all 74 WSI 
- *2 hours* : Run prediction pipeline on CAMELYON17 data for validation

## 14 Dec 2022
- *3 hours* : Supervisor meeting - noted in minutes

## 15 Dec 2022
- *2 hours* : Process prediction data to prepare survival modelling data. Calculate Malignancy score thresholded at 0.7 and mean intensity of each WSI for each patient for whom we have survival data from CBioPortal. 
- *2 hours* : Use features to fit a CPH model with two covariates and plot KM plots for each covariate at different levels. Plot each level's survival trend. Notice Mean intensity works very well with significant and realistic variations between each level whereas malignancy score thresholded at 0.7 gives the inverse relationship in its trends fore ach level (high malignancy score shows high survival and vice versa). Need to debug and understand the reason behind this inverse relationship but mean intensity produced very reasonable and effective survival trends for each level. 
- *0.5 hour* : Send results to supervisor for feedback 
- *2 hours* : Write status report and submit

## 16 Dec 2022
- *2 hours* : Based on feedback and my own observations Debug survival model code - Turns out fitting a single CPH model with multiple covariates creates the unusual relationship with subsequent covariates. The first covariate shows the expected result but not any other covariates. So, I trained two separate CPH models each with a single covariate only. One with mean intesnity and one with malignancy score. Now both CPH models produce the expected trends. Only difference being that Mean intensity baseline seems to be centered at 0.5 intensity with levels higher than 0.5 being below the baseline, and levels lower than 0.5 being above the baseline giving a more realistic expectation of survival trends with significantly large variations between levels. With malignancy score, the baseline seems to be centered at 0.0 such that all malignancy scores > 0.0 are under the baseline and with not much variation between levels, all trends of different levels being very close to each other, similar to the binary model thresholded at 0.5. This seems less realistic as the difference between 0.9 and 0.1 malignancy score should be much larger. This is likely because the general range in malignancy scores obtained is quite low, near 0, as the number of tiles with likelihood >=0.7 is quite low. This reduces the number of tiles being considered malignant significantly, and when calculated as a percentage of total tiles, this gives a very small number usually very close to 0. Thus the baseline is centered at 0 and the differences between levels are not significant as the model was fitted with very small values to begin with. 


## BREAK till 01 January 2023

## Week 12 - 10 hours

## 01 Jan 2023
- *1 hour* : Prepare dissertation template in repository
- *2 hours* : Read literature and complete the abstract of the dissertation

## 04 Jan 2023
- *2 hours* : Prepare citation manager in Mendeley and gather all literature to be used
- *3 hours* : Read exemplar dissertation from Hall of fame 2018

## TRAVELLING BACK TO GLASGOW ON 06 Jan 2023

## Week 13 - 6 hours

## 09 Jan 2023
- *3 hours* : Read exemplar dissertation from hall of fame 2019

## 11 Jan 2023
- *1 hour* : Supervisor meeting: 
  - Discuss dissertation structure - agree to follow template and fit our requirements to the given layout
  - Refer to dissertation #5 from 2019 hall of fame - Deep learning sample
  - Agree to finish Intro and background by next week
  - Intro can have aim, general problem and motivation
  - background will contain relevant literature used for the project, use all mendeley resources gathered
  - Briefly talk about survival time prediction results, median times are quite off compared to CBioPortal clinical data
  - agree that it is not essential to get the exact same prediction as we are not using any clinical data, there is no "wrong" prediction as long as Cox model shows significantly reasonable variations with different levels of the chosen covariate (mean intensity)
  - Can be explained in evaluation of viability of the whole project as there were lack of data, lack of clinical annotations etc which might all be affecting predictions. 
- *2 hours* : Refine and rewrite abstract of dissertation to make it more concise and to the point

## Week 14 - 25.5 hours
## 14 Jan 2023
- *2 hours* : Write introduction of dissertation - aim, motivation and general problem description

## 15 Jan 2023
- *8 hours* : Write background of dissertation - read and collect sources, write section 2.1 of background on Whole Slide Images. 
- *30 mins* : Plan to complete next background section on Deep Learning application in cancer detection and then Survival prediction from whole slide images.

## 16 Jan 2023
- *5 hours* : Write background section 2.2 on deep learning application and existing approaches to cancer detection using DL

## 17 Jan 2023
- *2 hours* : Supervisor meeting - Discuss analysis, design and implementation section of dissertation. Discuss survival results, agree results are significant for mean intensity with p=0.21. Discuss box plots to understand the spread of significant effect for each covariate. 

## 18 Jan 2023
- *2 hours* : Rewrite and refine introduction section based on supervisor discussion
- *2 hours* : Include macenko normalization in background for WSI

## 19 jan 2023
- *4 hours* : Write section 2.2 of background for data availability for deep learning model and start section 2.3 on prognosis prediction background

## Week 15 - 33 hours
## 21 Jan 2023 
- *8 hours* : Continue prognosis section 2.3 of background including existing survival time analysis methods and Cox proportional hazard model applications

## 22 Jan 2023
- *4 hours* : Write background on Kaplan Meier method and start analysis section 3.0

## 23 Jan 2023
- *8 hours* : Complete analysis section 3.1 explaining general problem in two stages (cancer detection and survival estimation). Complete section 3.2 on whole slide image application and choice of data for training (PCAM). 

## 24 Jan 2023
- *4 hours* : Write analysis section 3.3 on Processing - supervised learning and survival prediction process and image pre-processing

## 25 Jan 2023
- *2 hours* : Supervisor meeting - Discuss how to fit Design and implementation sections into project. Agree it is a bit inappropriate for research based projects. Discuss making extensions to project to try and get content to be included in implementation. Agree to not make major extensions this late. Discuss survival results including clinical data like Age, Sex etc as extensions - defeats the purpose of not using clinical data and testing the ability of histopathological images to characterize survival prognoses. Discuss including Hyperparameter optimization under implementation section. Agree to complete design section in the coming week and start on implementation next week. 
- *4 hours* : Make changes to section 3.1 amd 3.3.1 to include some more points on the general problem and approach to solve it. Complete section 3.3 - include ground truth validation and where ground truth will be sourced from and how the models will be tested and validated. 

## 26 Jan 2023
- *3 hours* : Design section 4.1 - general system overview for cancer detection. 

## Week 16 - 27 hours
## 28 Jan 2023
- *5 hours* : Design section 4.1.1 : tiling and pre-processing, 4.1.2 : tumour prediction - classiciation and segmentation 

## 29 Jan 2023
- *3 hours* : Design section 4.2 - 4.2.1 model covariates done

## 30 Jan 2023
- *4 hours* : Design section 4.2 - 4.2.2 survival time prediction done

## 1 Feb 2023
- *2 hours* : supervisor meeting - Discuss how to evaluate survival model. Bring up possibility of extending current project direction to include additional clinical covariates like Age in model. Agree to use predict_median as the survival time prediction. Agree that using multiple covariates gives a proportional model displaying hazard impact from all covariates invovled but plots dont display that since they are partial effect outcome plots which assume all other parameters remain unchanged. Discuss performing a cross-validation process to compare the two scores to see which gives better predictions. Cross-validate using jack-knifing where we train on 99 samples and leave 1 out for validation and repeat. Get accuracy value compared to ground truth labels for each model. Can be acceptable even with large errors as that serves as testament to the viability of given task. 

## 01 Feb 2023
- *5 hours* : Implementation section 5.1 - overview complete

## 02 Feb 2023
- *8 hours* : Implementation section 5.2 - Machine learning - 5.2.1 dataset and 5.2.2 preprocessing complete. 

## Week 17 - 21.5 hours
## 04 Feb 2023
- *5 hours* : Implementation section 5.2.3 - CNN architecture

## 05 Feb 2023
- *6 hours* : Implementation section 5.2.4 - Network Training and hyperparam tuning

## 06 Feb 2023
- *7 hours* : Hyperaparameter tuning - implement Ray[Tune] hyperparam tuning by modifying the existing network to accept variable params. Train models with 10k data points, and 50 epochs to test 20 random parameter combinations. Results - show tanh is best model with other parameters close to ones i used. 

## 08 Feb 2023
- *2 hours* : Start implementation section 5.3 - feature extraction and visualization section 5.3.1
- *1.5 hours* : Supervisor meeting - Show hyperparam tuning, agree its good and is sufficient since it agrees with my original model. Discuss plan for evaluation - Aim to complete classification evaluation by next week. Classification evaluation to contain hyperparam tuning results and calculate metrics (acc, precision, recall) for classified predictions and plot confusion matrix. Regression evaluation plan - 1. Discuss significance of derived scores as covariates with confidence interval plot. 2. For each model, plot predicted median survival against actual survival time to show if any relationship can be derived between the two. 3. Perform 5 fold cross validation to calculate RMSE scores for each model, plot SD and mean RMSE as distribution curves to show how predictions will be distributed.

## Week 18 - 29 hours
## 10 Feb 2023
- *6 hours* - Implementation section 5.3 - regression and feature extraction - complete feature extraction section 5.3.1 and model fitting section 5.3.2

## 11 Feb 2023
- *3 hours* : evaluation code for model - calculate and obtain metric values and confusion matrix from best 100k model. Try evaluate effects of varying dataset sizes on metrics to discuss increasing data potentially for better results if a trend can be observed. 
- *4 hours* : Write evaluation section 6.1  - Classification stage evaluation - 6.1.2 Metrics used and 6.1.3 how well does our DL model detect presence of cancer from histology images? 

## 12 Feb 2023
- *2 hours* : Perform 2nd round of hyperparameter optimization with fixed activation and batch size to find optimal LR and Dropout
- *4 hours* : Write evaluation section 6.1.1 - Hyperparameter optimization results

## 13 Feb 2023
- *8 hours* : Perform survival model evaluation - calculate 5 fold cross val RMSE scores and Std dev. Plot graph of prediction vs ground truth survival. Calculate effect of increasing data on prediction performance. 

## 15 Feb 2023
- *2 hours* : Supervisor meeting - discuss plan for evaluation. Agree that metrics calculated from survival evaluation look really good. Include 3 evaluations for prediction performance (RMSE+SD of 5fold val, prediction calibration and data increase effect). Add Evaluation for cox model using model statistics like Wald test, Log-rank test, regression coeff and hazard ratio + Conf Intervals. Plan to complete eval by next week

## Week 19 - 25 hours
## 18 Feb 2023
- *12 hours* : Write Evaluation metrics for survival model section 6.2 - Dissertation section 6.2.1 metrics for Cox evaluation and prediction performance evaluation (RMSE + SD) complete
## 19 Feb 2023
- *12 hours* : Complete first part of survival evaluation - Disseration section 6.2.2 how effective were our chosen image-based features as survival model covariates? 
- *5 hours* : plot CI graphs, caclulate metric scores for Cox model evaluation results
## 20 Feb 2023 - 23 Feb 2023
- *15 hours* : Complete evaluation and discussion sections
- Supervisor Meeting (22-02-2023):
  - Discuss cross validation evaluation of survival results
  - Discuss dataset size effect on survival results
  - Discuss RMSE + SD evaluation on survival results
  - Agree to finish evaluation and submit for first feedback

## Week 20
## 24 Feb 2023 - 2 March 2023:
- *15 hours* : Complete conclusion, future work. Make changes based on feedback received on first part of dissertation upto evaluation.
- Supervisor meeting (01-03-2023): Discuss feedback from the first half of the dissertation. Agree most changes are minor and looks very good. Change citation format to harvard. Add footnotes to some references. Agree to create an appendix to show result visualizations. Agree to move all code snippets to appendix to keep under 40 pages. Discuss validity of survival results and agree a conclusion of WSIs being inviable in predicting survival times is appropriate. Try find some literature to support it. 

## Week 21
## 3 March 2023 - 9 March 2023:
- *10 hours*: Complete appendix with images and results, finish conclusion with a citation of a work that had exact same conclusion as us (Wetsttein et al. 2019). Work on remaining feedback and finalize dissertation. 
- Supervisor meeting (08-03-2023): Discuss getting feedback on evaluation and conclusion next week. Agree appendix is appropriate and conclusion with the cited work is good. Discuss cleaning up the code base and getting ready for submission. 

## Week 22
## 10 March 2023 - 16 March 2023:
- *10 hours* - Clean up code base and make all python scripts executable with command line arguments rather than being notebooks. Add README and Manual files. Create submission style directory with proper organization. 
- *5 hours* - Receive complete feedback from supervisor, and work on making final fixes to dissertation. 
- Supervisor meeting (15-03-2023): Discuss final feedback and agree to finish dissertation and code base completion by this weekend for submission. Discuss video presentation and agree basic summary of the project with emphasis on results is appropriate for research based projects. 

## Total time = 402 hours