import os
import numpy as np
import matplotlib.pyplot as plt


for img in os.listdir("D:\PCAM DATA\WSI\Tiles\TCGA-A2-A0CU-01A"):
    img_path = os.path.join("D:\PCAM DATA\WSI\Tiles\TCGA-A2-A0CU-01A", img)
    image = plt.imread(img_path)
    if image.shape != (96,96,3):
        print(img)
    