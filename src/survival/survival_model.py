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

def read_data_prob(survival_path, tumour_path):
    # Read Survival clinical data 
    data = pd.read_csv(survival_path)
    
    # Calculate malignancy score of each Whole slide from its tumour predictions as a percentage of +ve tiles
    score = {}
    for pred_file in os.listdir(tumour_path):
        if pred_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(tumour_path, pred_file), index_col="Unnamed: 0")
            mean_intensity = df["predictions"].mean()
            
            pred_values = df['predictions']
            counts = pred_values[pred_values >= 0.7].count()
            malignant_score = counts/df.shape[0]
            
            score[pred_file[:12]] = [malignant_score]
            score[pred_file[:12]].append(mean_intensity)

    # Add calculated survival score into the survival data dataframe
    for id in score.keys():
        idx = data.index[data['Patient ID'] == id][0]
        data.loc[idx, "Malignancy Score"] = score[id][0]
        data.loc[idx, "Mean Intensity"] = score[id][1]
     
    # Remove any NaN values from dataframe for patients we dont have prediction data for since we only used WSI of deceased patients
    data["Malignancy Score"] = data["Malignancy Score"].replace(np.nan, 0)   
    data["Mean Intensity"] = data["Mean Intensity"].replace(np.nan, 0)

    # label fixing - Change DECEASED/ALIVE label to 1 (deceased) and 0 (alive)
    data["OS_STATUS"] = np.where(data["OS_STATUS"] == "1:DECEASED", 1, 0)

    # Drop empty rows which have no survival time available
    data.dropna(axis=0, how='any', inplace=True)
    
    return data 

# %%
temp = pd.read_csv("D:/PCAM DATA/Prediction_data/full_perdiction_set/TCGA-A2-A0CS-01Z-00-DX1.3986B545-63E8-4727-BCC1-701DE947D1FB-predictions.csv", index_col="Unnamed: 0")
counts = temp['predictions'].value_counts()
print(counts[1])
print(temp.shape[0])
print(counts[1]/temp.shape[0])

#%%
temp2 = pd.read_csv("D:/PCAM DATA/Prediction_data/probability_predictions_all/TCGA-A2-A0CS-01Z-00-DX1.3986B545-63E8-4727-BCC1-701DE947D1FB-predictions.csv", index_col="Unnamed: 0")
col = temp2['predictions']
val_count = col[col >= 0.7].count()
print(val_count)
print(temp2.shape[0])
print(val_count/temp2.shape[0])

#%%
data = pd.read_csv("D:/PCAM DATA/Survival/survival_data.csv")
data.head()

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

def cox_model_score(data):
    data = data[['OS_MONTHS', 'OS_STATUS', 'Malignancy Score']]
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
    
    return data, cph

def cox_model_intensity(data):
    data = data[['OS_MONTHS', 'OS_STATUS', 'Mean Intensity']]
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
    
    return data, cph

#%%
# if __name__ == "__main__":
# args = parser.parse_args()

pred_dir = "D:/PCAM DATA/Prediction_data/full_perdiction_set"
pred_dir_prob = "D:/PCAM DATA/Prediction_data/probability_predictions_all"
data_path ="D:/PCAM DATA/Survival/survival_data.csv"

data = read_data(data_path, pred_dir)
data_prob = read_data_prob(data_path, pred_dir_prob)

# %%
data =  data[data["Malignancy Score"] != 0.0]
data.head()

#%%
data_prob = data_prob[data_prob["Malignancy Score"] != 0.0]
data_prob.head()

#%%
cox_data, cox = cox_model(data)
plt.subplots(figsize=(10, 6))
cox.plot()

#%%
cox_data_score, cox_score = cox_model_score(data_prob)
plt.subplots(figsize=(10, 6))
cox_score.plot()

cox_data_intensity, cox_intensity = cox_model_intensity(data_prob)
plt.subplots(figsize=(10, 6))
cox_intensity.plot()

# %%
cox.plot_partial_effects_on_outcome(covariates='Malignancy Score', values=[
    0.2, 0.3, 0.4, 0.8, 0.9
    ], cmap='coolwarm')
plt.title("Binary 0|1 prediction model - Malignancy Score covariate")

#%%
cox_intensity.plot_partial_effects_on_outcome(covariates='Mean Intensity', values=[
    0.48, 0.42, 0.28, 0.9
    ], cmap='coolwarm')
plt.title("Probability prediction model - Mean Intensity covariate")

#%%
cox_score.plot_partial_effects_on_outcome(covariates='Malignancy Score', values=[
    0.1, 0.42, 0.28, 0.9
    ], cmap='coolwarm')
plt.title("Probability prediction model - Malignancy Score covariate")

# %%
# results = proportional_hazard_test(cox, cox_data, time_transform='rank')
# results.print_summary(decimals=3, model="untransformed variables")

# %%
results_prob = proportional_hazard_test(cox_intensity, cox_data_intensity, time_transform='rank')
results_prob.print_summary(decimals=3, model="untransformed variables")

# %%
def calc_score(path):
    df = pd.read_csv(path)
    
    pred_values = df['predictions']
    counts = pred_values[pred_values >= 0.7].count()
    malignant_score = counts/df.shape[0]
    
    return malignant_score

# %%
print(calc_score("D:/PCAM DATA/Prediction_data/probability_predictions_all/TCGA-GM-A2DA-01Z-00-DX1.6E409F88-F654-48C8-A753-BB054037BE16-predictions.csv"))

# %%
