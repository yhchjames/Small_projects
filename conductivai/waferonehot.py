#%%
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import xgboost as xgb
# from sklearn.metrics import accuracy_score  # 準確率
from xgboost import plot_importance


# %%
data = pd.read_csv("test_assignment_sim.csv")


pd.options.display.max_columns = 200
#%%

colist = list(data.columns)

#%%
data.head(100)

# drawdata = data.iloc[:,3:6]
# drawdata.plot()

#%%
data.info()
#%%

sites_datas = []
for i in range(49):
    slice = data.iloc[:,[0,1,2,3,i+4]]
    slice['thickness'] = slice.iloc[:,4]
    slice.iloc[:,4] = True
    sites_datas.append(slice)



#%%
site_thickness = sites_datas[0]

for j in range(1,49):
    site_thickness = pd.concat([site_thickness,sites_datas[j]])

site_thickness



#%%
colist +=['thickness']
colist

#%%
site_thickness = site_thickness[colist]

#%%
site_thickness.head()
# %%

site_thickness.fillna(0,inplace=True)
site_thickness



# %%
site_thickness.reset_index(drop=True)
# %%
import lightgbm
# %%

# %%

# %%

# %%

# %%
