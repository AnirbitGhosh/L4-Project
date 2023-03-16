# User manual 

This contains basic instructions on how to use the key files in our code base. 

<span style="color: red">
IMPORTANT NOTE: </span> Whole Slide Image data could not be provided here with the source code due to its size exceeding 1 TB. Processing a single WSI through the preporcessor to generate normalized tiles, then through the classification model to generate tile-level predictions takes over 40 mins per image. 

</br></br>
### Data directory structure 

For training our model we need the data to be organized in the following structure inside a root directory:
```
- root
    |_ train
        |_ tiles
            |_ .jpeg tile images
        |_ train_labels.csv
    |_ test
        |_ tiles
            |_ .jpeg tile images
        |_ test_labels.csv
    |_ validation
        |_ tiles
            |_ .jpeg tile images
        |_ validation_labels.csv
```
</br>
For WSIs to be processed using our trained model for feature extraction, we require the following structure inside a root directory:

```
- root
    |_ Whole Slide Images
        |_ .svs format slide images, named with TCGA ID
    |_ Tiles
        |_ <TCGA ID of WSI>
            |_ .tif format tiles
        |_ <TCGA ID of WSI>
        .
        .
        .
```


Pre-processing instructions:
- Tiling:
  - Run following command to execute `/preprocessing/tile-slide.py`:
  ```
  python tile-slide.py -i <PATH to directory containing whole slide images> -o <PATH to output directory to save tiled images in> -s <SIZE of tiles [default=96]>
  ```
  The output will be folders containing macneko normalized tiles of each WSI in the root directory. Each tile directory will be named with the WSI TCGA ID (shown in folder structure above).

</br>
Training instrctions 

- Training model:
  - Run following command to train a CNN using `/models/model_01.py`
  ```
  python models_01.py -p <PATH to root data directory> -o <PATH to save trained model, with filename ending in .pt>
  ```
  The output will be a trained model, trained on the input tiled data provided. 

</br>
Post-processing instructions

- Generate cancer tumour predictions on tiled WSIs:
  - Run following command to generate malignancy predictions using `/data_processing/generate_predictions.py`:
  ```
  python generate_predictions.py -n <PATH to trained model weights .pt file> -t <Pass directory containing all tile directories of WSIs to be predicted> -o <PATH to output directory where each WSI's predictions will be saved as a .csv> -w <PATH to directory containing .svs whole slide images>
  ```
This will generate tile-level malignancy predictions for each WSI. The predictions will be stored in a .csv format, named as the TCGA ID of the corresponding WSI. In the CSV file,  predictions are stored against each tile's coordinate name. 

- Generate tumour map visualization:
  - Run following command to produce an image representation of the model generated tumour predictions for WSIs using `/data_processing/visualize_tiles.py`:
  ```
  python visualize_tiles.py -i <PATH to directory containing .svs format Whole Slide Images for which predictions have been generated> -l <PATH to directory containing all .csv files of generated WSI predictions>
  ```
  This will produce .png format images (heatmaps or binary maps) showing the predicted tumour regions of each given WSI and their corresponding prediction results obtained from our trained model. 

