#%%
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import proportional_hazard_test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)
    
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pred', type=dir_path, help='Pass directory containing csv files of tumour predictions', default="D:/PCAM DATA/Survival/tumour_predictions")
parser.add_argument ('-d', '--data', help='Pass path to survival data csv', default="D:/PCAM DATA/Survival/survival_data.csv")

def read_data(survival_path, tumour_path):
    # Read Survival clinical data 
    data = pd.read_csv(survival_path)
    
    # Calculate malignancy score of each Whole slide from its tumour predictions as a percentage of +ve tiles
    score = {}
    for pred_file in os.listdir(tumour_path):
        if pred_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(tumour_path, pred_file), index_col="Unnamed: 0")
            counts = df['predictions'].value_counts()
            malignant_score = counts[1]/len(df)
            score[pred_file[:12]] = malignant_score

    # Add calculated survival score into the survival data dataframe
    for id in score.keys():
        idx = data.index[data['Patient ID'] == id][0]
        data.loc[idx, "Malignancy Score"] = score[id]
     
    # Remove any NaN values from dataframe for patients we dont have prediction data for since we only used WSI of deceased patients
    data["Malignancy Score"] = data["Malignancy Score"].replace(np.nan, 0)   

    # label fixing - Change DECEASED/ALIVE label to 1 (deceased) and 0 (alive)
    data["OS_STATUS"] = np.where(data["OS_STATUS"] == "1:DECEASED", 1, 0)

    # Drop empty rows which have no survival time available
    data.dropna(axis=0, how='any', inplace=True)
    
    return data 

#%%
def KM_plot(data):
    T = data["OS_MONTHS"]
    E = data["OS_STATUS"]
    
    kmf = KaplanMeierFitter()
    kmf.fit(durations= T, event_observed = E)
    
    kmf.plot_survival_function()
    plt.title("Survival Plot")  

#%%
def cox_model(data):
    data = data[['OS_MONTHS', 'OS_STATUS', 'Malignancy Score']]
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
    
    return data, cph

#%%
# if __name__ == "__main__":
# args = parser.parse_args()

pred_dir = "D:/PCAM DATA/Survival/tumour_predictions"
data_path ="D:/PCAM DATA/Survival/survival_data.csv"

data = read_data(data_path, pred_dir)

# %%
data =  data[data["Malignancy Score"] != 0.0]
data.head()

# %%
cox_data, cox = cox_model(data)
plt.subplots(figsize=(10, 6))
cox.plot()

# %%
cox.plot_partial_effects_on_outcome(covariates='Malignancy Score', values=[
    0.1
    ], cmap='coolwarm')

# %%
results = proportional_hazard_test(cox, cox_data, time_transform='rank')
results.print_summary(decimals=3, model="untransformed variables")

# %%
