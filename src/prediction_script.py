import os

os.chdir("preprocessing")
os.system('python tile-slide.py -i "D:/PCAM DATA/WSI/Whole Slide Images" -o "D:/PCAM DATA/WSI/downsampled_tiles" -s 96')


os.chdir("../data_processing")
os.system('python generate_predictions.py -n "D:/PCAM DATA/trained_models/weights_01_100k.pt" -t "D:/PCAM DATA/WSI/downsampled_tiles" -o "D:/PCAM DATA/Prediction_data/downsampled_predictions" -w "D:/PCAM DATA/WSI/Whole Slide Images"')
os.system('python visualize_tiles.py -i "D:/PCAM DATA/WSI/Whole Slide Images" -l "D:/PCAM DATA/Prediction_data/downsampled_predictions"')
