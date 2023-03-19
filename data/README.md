# Data used in the project

## PCAM data
The annotated, tiled data used to train our machine learning model to classify breast metastatic or benign tissue. The dataset is too large to upload as it contains over 300,000 images totaling over 15GB across training, validation and test sets. Can be downloaded from [PCAM Github](https://github.com/basveeling/pcam). We have added the annotation data for each dataset as a `.csv` file in each directory. It contains malignant or benign annotation as a binary label against each tile. 

## TCGA data
The 74 WSI dataset used to extract features that serve as covariates in survival modeling. Each WSI is over 1.5GB in size, totaling over 80GB for all 74 image files. They could not be uploaded due to their size. Can be downloaded from [TCGA repoistory](https://portal.gdc.cancer.gov/repository?filters=%7B%22op%22%3A%22and%22%2C%22content%22%3A%5B%7B%22content%22%3A%7B%22field%22%3A%22files.cases.primary_site%22%2C%22value%22%3A%5B%22breast%22%5D%7D%2C%22op%22%3A%22in%22%7D%2C%7B%22op%22%3A%22in%22%2C%22content%22%3A%7B%22field%22%3A%22files.data_type%22%2C%22value%22%3A%5B%22Slide%20Image%22%5D%7D%7D%5D%7D). The IDs of each WSI used can be found in the `/TCGA data/TCGA_data_ID.csv` file. 

## Survival data
The corresponding clinical data for each of the 74 TCGA samples was retrieved from CBioPortal. It is recorded in `/Survival data/survival_data.csv`. The survival duration in months is under `OS_MONTHS`, survival status (deceased, alive) under `OS_STATUS` and patient ID under `Patient ID`. 

## Prediction results
`/Prediction results/binary_predictions` contains a `.csv` file and a `.png` image for each of the 74 WSIs. The `.csv` file contains discrete tile-level prediction results generated by our model, as a 0 (benign) or 1 (malignant) class label. The `.png` image is a binary segmentation map showing the malignant and benign regions according to the binary prediction results. 
`/Prediction results/probability_predictions` contains similar files for each of the 74 WSIs. Instead of discrete, binary class label predictions, this directory contains tile-level malignant probability (probability of a tile belonging to the malignant class) predictions for each WSI. The `.png` image is a continuous heatmap showing the malignant probability distribution of each WSI. 

