# %% 
import pandas as pd
import numpy as np
# %%
data = pd.read_csv("../../data/raw/bank.csv", sep = ';')
model = pd.read_pickle("../../models/classification_model.pkl")
# %%
X = data[model['features']]
# %%
y_pred = model['model'].predict(X)
y_pred_proba = model['model'].predict_proba(X)
# %%
data['predSuccess'] = y_pred
data['probSucces'] = y_pred_proba[:,1]
data.to_excel("../../data/processed/model_predictions.xlsx", index = False)
# %%
data.head()