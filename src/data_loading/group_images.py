from email.mime import base
import os 
import shutil
import pandas as pd

base_path = "D:/PCAM DATA"

train_path = os.path.join(base_path, "train/tiles")
test_path = os.path.join(base_path, "test/tiles")
val_path = os.path.join(base_path, "validation/tiles")

train_df = pd.read_csv(os.path.join(base_path, "train/train_labels.csv"))
test_df = pd.read_csv(os.path.join(base_path, "test/test_labels.csv"))
val_df = pd.read_csv(os.path.join(base_path, "validation/validation_labels.csv"))

# print(len(os.listdir(train_path)))

for file in os.listdir(train_path):
    if file.endswith('.jpeg'):
        img_class = train_df.loc[int(file[:-5])].values[3]
        if img_class == 1:
            shutil.move(train_path+"/"+file, train_path+"/1/"+file)
        else:
            shutil.move(train_path+"/"+file, train_path+"/0/"+file)
        print(f"moved {file} to {img_class}")
        
for file in os.listdir(val_path):
    if file.endswith('.jpeg'):
        img_class = val_df.loc[int(file[:-5])].values[3]
        if img_class == 1:
            shutil.move(val_path+"/"+file, val_path+"/1/"+file)
        else:
            shutil.move(val_path+"/"+file, val_path+"/0/"+file)
        print(f"moved {file} to {img_class}")