# L4-Project - Breast cancer survival analysis using deep learning

## Code structure
- `/data_loading` : Contains code to prepare raw data before being processing through preprocessing pipeline and the CNN. `data_transforms.py` contains all relevant transformers for data augmentation. `dataset.py` contains the custom `BreastCancerDataset` class which is used to load images and associated labels into a streamlined object for easier retrieval by torch dataloader. `extract_data.py` is used to extract image arrays from the PCAM dataset's HDF5 file and store the images as `.jpeg` in a given directory. 

- `/data_processing` : Contains code to obtain bulk predictions from input whole slide image tiles in `generate_predictions.py`, predictions are stored as a csv file with a predicted value against each tile name. `visualize_tiles.py` generates binary and continuous segmenation maps to visualize malignant areas in a given tissue. 

- `/models` : `custom_network.py` contains our CNN implementation and training functions. `evaluation.py` was used to generate performance metrics and confusion matrices of our trained model on a testing dataset. This was originally written as `.ipynb` notebook and is intended to be executed one cell at a time to observe outputs produced at each step but it has been converted into a `.py` file for execution through command lines which will produce all the outputs at once. `hyperparam_tune.py` is our implemenation of a `Ray[Tune]` pipeline to optimize hyperparameters of a given model. `model_01.py` contains the actual code where we have loaded our image data into `BreastCancerDataset` objects and trained and saved our model. `predictor.py` was a test file written to debug and obtain model predictions. `utils.py` contains training utility functions that used by the main training function. 

- `/preprocessing` : Contains code to macenko normalize images in `macenko_norm.py`. `pixel_analysis.py` was a test script made to calculate mean background intensity and standard deviation of images to set the right threshold to eliminate blank images. `tile-slide.py` is the main file to split an input Whole Slide image into 96x96px tiles and save each tile as a .tiff file. 

- `/survival` : `survival_model.py` contains the code to set up Cox hazard models using the CNN generated predictions and patient survival data in .csv format. The code fits the model with the data and predicts median survival time for a given covariate level. Contains statistical test results on the hazard models. `evaluation.py` contains implementations of 5-fold cross validation, RMSE calculation of survival models and the investigation of dataset size on survival prediction performance.

## Build requirements
Core requirements:
- `Python` >= 3.9
- `OpenSlide` (with win64/Linux binaries)
- `PyTorch` = 1.13.1 (py3.9_cuda11.7_cudnn8_0)
- `PyTorch-cuda` = 11.7
- `ray` = 2.2.0
- `Lifelines` >= 0.27.4
- `h5py` >= 3.7.0
- `scikit-learn` >= 1.0.2
- `scipy` >= 1.7.3
- `numpy` >= 1.22.4
- `openCV` >= 4.5.5
- `matplotlib`
- `seaborn`
- CUDA 11.7 supported GPU

Additional requirements (conda format) : [requirements.txt](../requirements.txt)

Conda environment setup : [environment.yml](../environment.yml)

## Environment setup
Create a virtual environment with relevant dependencies using the command:

- `conda env create -f environment.yml`

Additional steps:
- Might need to separately download and install PyTorch and CUDA, use the commands provided below:
    - Linux : `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
    - Windows : `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
    - MacOS : PyTorch CUDA is NOT supported 
  
- Might need to set up OpenSlide separately:
  
  WINDOWS users (for other OS, check [https://openslide.org/](https://openslide.org/) for instructions):
  -  Run `pip install openslide-python`
  -  Go to [`https://openslide.org/download/`](https://openslide.org/download/) and download latest OpenSlide windows binaries (we used 64 bit binaries release data: 2022-12-17).
  -  Locate `\lib\site-packages\openslide` directory inside the created anaconda environment directory 
  -  Open `lowlevel.py` file 
  -  At the top of the file, under the `from __future__ import division` line, add the following code:
```python
from __future__ import division

import os
os.environ['PATH'] = <PATH TO bin DIRECTORY INSIDE DOWNLOADED BINARIES> 
```





