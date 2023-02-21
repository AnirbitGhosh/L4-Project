#%%
import sys
import os
sys.path.insert(0, os.path.abspath("C:\\Users\Anirbit\\L4 Project\\L4-Project\\src"))

from survival.survival_model import read_data, read_data_prob, cox_model, cox_model_intensity
from matplotlib import pyplot as plt
import numpy as np
import sklearn.metrics as sk
import math
import statistics

#%%
pred_dir = "D:/PCAM DATA/Prediction_data/full_perdiction_set"
pred_dir_prob = "D:/PCAM DATA/Prediction_data/probability_predictions_all"
data_path ="D:/PCAM DATA/Survival/survival_data.csv"

data_binary = read_data(data_path, pred_dir)
data_prob = read_data_prob(data_path, pred_dir_prob)

# %%
data_binary =  data_binary[data_binary["Malignancy Score"] != 0.0]
print(data_binary.head())

data_prob = data_prob[data_prob["Mean Intensity"] != -1]
print(data_prob.head())

# %%
#### Checking trend between predicted and actual survival ######
cox_binary_data, cox_binary = cox_model(data_binary)
cox_intensity_data, cox_intensity = cox_model_intensity(data_prob)

# %%
data_binary['median_prediction'] = cox_binary.predict_median(cox_binary_data)
print(data_binary.head())
data_prob['median_prediction'] = cox_intensity.predict_median(cox_intensity_data)
print(data_prob.head())

# %%
prob_x = data_prob["OS_MONTHS"]
prob_y = data_prob["median_prediction"]
prob_a, prob_b = np.polyfit(prob_x, prob_y, 1)

plt.scatter(prob_x, prob_y, label="Survival time")
plt.plot(prob_x, prob_a*prob_x+prob_b, 'r', label="Linear best fit")
plt.xlabel("Actual survival duration (months)")
plt.ylabel("Predicted median survival time (months)")
plt.yticks(np.linspace(0, 200, 5))
plt.title("Predicted vs actual survival time - MMI covariate")
plt.legend()
plt.savefig("../../dissertation/images/pva-mmi.png", format='png')

# %%
bin_x = data_binary["OS_MONTHS"]
bin_y = data_binary["median_prediction"]
bin_a, bin_b = np.polyfit(bin_x, bin_y, 1)

plt.scatter(bin_x, bin_y, label="Survival time")
plt.plot(bin_x, bin_a*bin_x+bin_b, 'r', label="Linear best fit")
plt.xlabel("Actual survival duration (months)")
plt.ylabel("Predicted median survival time (months)")
plt.yticks(np.linspace(0, 200, 5))
plt.title("Predicted vs actual survival time - MSS covariate")
plt.legend()
plt.savefig("../../dissertation/images/pva-mss.png", format='png')

#%%
######## Dataset size effect ######
def calculate_rmse(train, valid, cox_model):
    cox_fold_data, cox_fold = cox_model(train) 
    predictions = cox_fold.predict_median(valid)

    actual = valid["OS_MONTHS"]
    rmse = math.sqrt(sk.mean_squared_error(actual, predictions))
    return rmse

train_data_binary = data_binary
train_data_prob = data_prob

#%%
#### BINARY DATA
### 25% dataset
len_25 = len(train_data_binary) // 4
train_data_25 = train_data_binary[:len_25]
binary_25_rmse = calculate_rmse(train_data_25, data_binary, cox_model)
print("RMSE 25% binary training data: ", binary_25_rmse)

### 50% dataset
len_50 = len(train_data_binary) // 2
train_data_50 = train_data_binary[:len_50]
binary_50_rmse = calculate_rmse(train_data_50, data_binary, cox_model)
print("RMSE 50% binary training data: ", binary_50_rmse)

### 75%
len_75 = (len(train_data_binary) // 4)*3
train_data_75 = train_data_binary[:len_75]
binary_75_rmse = calculate_rmse(train_data_75, data_binary, cox_model)
print("RMSE 75% binary training data: ", binary_75_rmse)

### 100%
train_data_100 = train_data_binary
binary_100_rmse = calculate_rmse(train_data_100, data_binary, cox_model)
print("RMSE 100% binary training data: ", binary_100_rmse)


#%%
#### PROB DATA
### 25% dataset
len_25 = len(train_data_prob) // 4
train_data_25 = train_data_prob[:len_25]
prob_25_rmse = calculate_rmse(train_data_25, data_prob, cox_model)
print("RMSE 25% prob training data: ", prob_25_rmse)

### 50% dataset
len_50 = len(train_data_prob) // 2
train_data_50 = train_data_prob[:len_50]
prob_50_rmse = calculate_rmse(train_data_50, data_prob, cox_model)
print("RMSE 50% prob training data: ", prob_50_rmse)

### 75%
len_75 = (len(train_data_prob) // 4)*3
train_data_75 = train_data_prob[:len_75]
prob_75_rmse = calculate_rmse(train_data_75, data_prob, cox_model)
print("RMSE 75% prob training data: ", prob_75_rmse)

### 100%
train_data_100 = train_data_prob
prob_100_rmse = calculate_rmse(train_data_100, data_prob, cox_model)
print("RMSE 100% prob training data: ", prob_100_rmse)

#%%
x = [25, 50, 75, 100]
y1 = [prob_25_rmse, prob_50_rmse, prob_75_rmse, prob_100_rmse]
y2 = [binary_25_rmse, binary_50_rmse, binary_75_rmse, binary_100_rmse]

plt.figure()
plt.plot(x, y1, label="Mean malignant intensity")
plt.plot(x, y2, label="Malignancy spread score")
plt.xticks(x)
plt.xlabel("Training data size as % of total dataset")
plt.ylabel("RMSE (months)")
plt.title("RMSE vs training data size")
plt.legend()

# %%
###### 5 fold cross validation ########
#### Probability Data
prob_fold_1 = data_prob[:15]
prob_fold_2 = data_prob[15:30]
prob_fold_3 = data_prob[30:45]
prob_fold_4 = data_prob[45:60]
prob_fold_5 = data_prob[60:]

def calculate_rmse(train, valid, cox_model):
    cox_fold_data, cox_fold = cox_model(train) 
    predictions = cox_fold.predict_median(valid)

    actual = valid["OS_MONTHS"]
    rmse = math.sqrt(sk.mean_squared_error(actual, predictions))
    
    return rmse

#%%
#### FOLD 1
train_data =  prob_fold_2.append([prob_fold_3, prob_fold_4, prob_fold_5])
valid_data1 = prob_fold_1

fold1_rmse = calculate_rmse(train_data, valid_data1, cox_model_intensity)
print("Fold 1 RMSE SCORE : ", fold1_rmse)

# %%
#### FOLD 2
train_data =  prob_fold_1.append([prob_fold_3, prob_fold_4, prob_fold_5])
valid_data2 = prob_fold_2

fold2_rmse = calculate_rmse(train_data, valid_data2, cox_model_intensity)
print("Fold 2 RMSE SCORE : ", fold2_rmse)

# %%
#### FOLD 3
    
train_data =  prob_fold_1.append([prob_fold_2, prob_fold_4, prob_fold_5])
valid_data3 = prob_fold_3

fold3_rmse = calculate_rmse(train_data, valid_data3, cox_model_intensity)
print("Fold 3 RMSE SCORE : ", fold3_rmse)

# %%
#### FOLD 4
train_data =  prob_fold_1.append([prob_fold_2, prob_fold_3, prob_fold_5])
valid_data4 = prob_fold_4

fold4_rmse = calculate_rmse(train_data, valid_data4, cox_model_intensity)
print("Fold 4 RMSE SCORE : ", fold4_rmse)

# %%
#### FOLD 5
train_data =  prob_fold_1.append([prob_fold_2, prob_fold_3, prob_fold_4])
valid_data5 = prob_fold_5

fold5_rmse = calculate_rmse(train_data, valid_data5, cox_model_intensity)
print("Fold 5 RMSE SCORE : ", fold5_rmse)

# %%
intensity_mean_rmse = statistics.mean([fold1_rmse, fold2_rmse, fold3_rmse, fold4_rmse, fold5_rmse])
intensity_sd_rmse = statistics.stdev([fold1_rmse, fold2_rmse, fold3_rmse, fold4_rmse, fold5_rmse])
print("Average RMSE across 5 folds when using intensity score: ", intensity_mean_rmse)
print("Std Dev across 5 folds when using intensity score: ", intensity_sd_rmse)

# %%
###### 5 fold cross validation ########
#### Binary Data
bin_fold_1 = data_binary[:15]
bin_fold_2 = data_binary[15:30]
bin_fold_3 = data_binary[30:45]
bin_fold_4 = data_binary[45:60]
bin_fold_5 = data_binary[60:]


#%%
#### FOLD 1
train_data =  bin_fold_2.append([bin_fold_3, bin_fold_4, bin_fold_5])
valid_data1 = bin_fold_1

fold1_rmse = calculate_rmse(train_data, valid_data1, cox_model)
print("Fold 1 (binary) RMSE SCORE : ", fold1_rmse)

# %%
#### FOLD 2
train_data =  bin_fold_1.append([bin_fold_3, bin_fold_4, bin_fold_5])
valid_data2 = bin_fold_2

fold2_rmse = calculate_rmse(train_data, valid_data2, cox_model)
print("Fold 2 (binary) RMSE SCORE : ", fold2_rmse)

# %%
#### FOLD 3
train_data =  bin_fold_1.append([bin_fold_2, bin_fold_4, bin_fold_5])
valid_data3 = bin_fold_3

fold3_rmse = calculate_rmse(train_data, valid_data3, cox_model)
print("Fold 3 (binary) RMSE SCORE : ", fold3_rmse)

# %%
#### FOLD 4
train_data =  bin_fold_1.append([bin_fold_2, bin_fold_3, bin_fold_5])
valid_data4 = bin_fold_4

fold4_rmse = calculate_rmse(train_data, valid_data4, cox_model)
print("Fold 4 (binary) RMSE SCORE : ", fold4_rmse)

# %%
#### FOLD 5
train_data =  bin_fold_1.append([bin_fold_2, bin_fold_3, bin_fold_4])
valid_data5 = bin_fold_5

fold5_rmse = calculate_rmse(train_data, valid_data5, cox_model)
print("Fold 5 (binary) RMSE SCORE : ", fold5_rmse)

# %%
score_mean_rmse = statistics.mean([fold1_rmse, fold2_rmse, fold3_rmse, fold4_rmse, fold5_rmse])
score_sd_rmse = statistics.stdev([fold1_rmse, fold2_rmse, fold3_rmse, fold4_rmse, fold5_rmse])
print("Average RMSE across 5 folds when using binary malignancy score: ", score_mean_rmse)
print("Std Dev across 5 folds when using binary malignancy score: ", score_sd_rmse)
