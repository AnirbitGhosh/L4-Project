#%%
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import proportional_hazard_test
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

#%%
survival_df = pd.read_csv("D:/PCAM DATA/Survival/survival_data.csv")
survival_df.head()

# %%
score = {}
pred_path = "D:/PCAM DATA/WSI/final_predictions/"
for prediction in os.listdir(pred_path):
    if prediction.endswith(".csv"):
        df = pd.read_csv(os.path.join(pred_path, prediction), index_col="Unnamed: 0")
        counts = df['predictions'].value_counts()
        malignant_score = counts[1]/len(df)
        score[prediction[:12]] = malignant_score

# %%
print(score)

# %%
for id in score.keys():
    idx = survival_df.index[survival_df['Patient ID'] == id][0]
    survival_df.loc[idx, "Malignancy Score"] = score[id]

survival_df["Malignancy Score"] = survival_df["Malignancy Score"].replace(np.nan, 0)

# %%
# label fixing
survival_df["OS_STATUS"] = np.where(survival_df["OS_STATUS"] == "1:DECEASED", 1, 0)
counts = survival_df["OS_STATUS"].value_counts()
print(counts)

# %%
survival_df.head()

# %%
# Drop empty rows
survival_df.dropna(axis=0, how='any', inplace=True)
survival_df.isnull().sum()

counts = survival_df["OS_STATUS"].value_counts()
print(counts)

# %%
survival_df = survival_df[survival_df["Malignancy Score"] != 0.0]
survival_df.head()

########################################################
########################################################

# %%
#KM plot
T = survival_df["OS_MONTHS"]
E = survival_df["OS_STATUS"]
kmf = KaplanMeierFitter()
kmf.fit(durations= T, event_observed = E)
kmf.plot_survival_function()
plt.title("Survival Plot")

# %%
median_ = kmf.median_survival_time_
median_confidence_interval = median_survival_times(kmf.confidence_interval_)

print(median_)
print(median_confidence_interval)

# %%
## Cox model
data = survival_df[['OS_MONTHS', 'OS_STATUS', 'Malignancy Score']]
data.head()

# %%
cph = CoxPHFitter()
cph.fit(data, duration_col='OS_MONTHS', event_col='OS_STATUS')
cph.print_summary()

# %%
plt.subplots(figsize=(10, 6))
cph.plot()

# %%
cph.plot_partial_effects_on_outcome(covariates='Malignancy Score', values=[
    0.1, 0.5, 0.75, 0.9
    ], cmap='coolwarm')

# %%
results = proportional_hazard_test(cph, data, time_transform='rank')
results.print_summary(decimals=3, model="untransformed variables")

# %%
