#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from itertools import *
import pytest
import time

#%%
def test_datasetting():
    data = pd.read_csv("test_assignment_sim.csv")
    site_co = pd.read_csv("site_coordinates.csv")
    site_co

    #%%
    site_co.info()

    #%%

    sites_datas = []
    for i in range(49):
        slice = data.iloc[:,[0,1,2,3]].copy()
        slice['thickness'] = data.iloc[:,i+4]
        slice['SITE'] = i # in lgb, catagory should be a int
        slice['S_X'] = site_co['SITE_X'][i]
        slice['S_Y'] = site_co['SITE_Y'][i]
        sites_datas.append(slice)

    # %%
    site_thickness = sites_datas[0]
    for j in range(1,49):
        site_thickness = pd.concat([site_thickness,sites_datas[j]])

    site_thickness

    #%%

    clist = list(site_thickness.columns)
    clist.append(clist.pop(4))
    clist.pop(4)
    clist


    # %%
    site_thickness = site_thickness[clist]
    site_thickness
    # %%
    site_thickness.reset_index(drop=True)

    # %%
    # site_thickness['TOOL']= site_thickness['TOOL'].astype('object')
    site_thickness.info()
    #%%
    site_thickness.describe()

    # %%
    site_thickness['TOOL'] = site_thickness['TOOL'].astype('category')
    site_thickness.info()
    #%%

    site_thickness.describe()
# %%

# def test_datasplit():
    dftrain,dftest = train_test_split(site_thickness,test_size=0.1)

    categorical_features = ['TOOL']
    lgb_train = lgb.Dataset(dftrain.drop(['thickness'],axis = 1),label=dftrain['thickness'],
                            categorical_feature = categorical_features,free_raw_data=False)

    lgb_valid = lgb.Dataset(dftest.drop(['thickness'],axis = 1),label=dftest['thickness'],
                            categorical_feature = categorical_features,
                            reference=lgb_train,free_raw_data=False)
# %%
# def test_trainlgb():
    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 250,
        'learning_rate': 0.05,
        # 'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'verbose': 0,
        'max_depth':25
    }
    # %%
    gbm = lgb.train(params,lgb_train,categorical_feature=categorical_features)

    gbm.save_model('lgbmodel.txt')

# def test_predictmodel():
    model = lgb.Booster(model_file='lgbmodel.txt')
    


    y_pred = model.predict(dftest.drop(['thickness'],axis = 1), num_iteration=model.best_iteration) 
    from sklearn.metrics import mean_squared_error
    print('The rmse of prediction is:', mean_squared_error(dftest['thickness'], y_pred) ** 0.5)
    lgb.plot_importance(model)
# %%




#%%


'''
estimator = lgb.LGBMRegressor(boosting_type= 'gbdt',objective= 'regression',metric='rmse',learning_rate=0.1)

parameters = {
              'max_depth': [25,35,45],
              # 'device_type':'gpu',
              'num_leaves':[150,200,250],
            #   'learning_rate': [0.01, 0.1, 0.15],
            #   'feature_fraction': [0.6, 0.8, 0.9],
              'bagging_fraction': [0.9],
              'bagging_freq': [2]
            #   'lambda_l1': [0, 0.1,0.5, 0.6],
            #   'lambda_l2': [0, 15, 35],
            #   'cat_smooth': [1, 15, 35]
}
start = time.time()

gbm = GridSearchCV(estimator, parameters, cv=3)
gbm.fit(dftrain.drop(['thickness'],axis = 1), dftrain['thickness'])
print ('Time for epoch CV is {} sec'.format(time.time()-start))
print('Best parameters found by grid search are:', gbm.best_params_)
print('Best score is :',gbm.best_score_)
'''
# %%
def test_check_outliear():
# Checking extrapolating 1 standard deviation outside of the ranges
    model = lgb.Booster(model_file='lgbmodel.txt')
    FF = [0.86,1.03]
    SP = [0.3293,0.368]
    DT = [64.563,77.81]
    TL = [1,2,3,4]
    SX = [147000,-147000,0]
    SY = [14700,-147000,0]

    exlist = []
    for i in product(FF,SP,DT,TL,SX,SY):
        exlist.append(i)    
    # %%
    exdf = pd.DataFrame(exlist,columns=['FF', 'SP','DT','TL','SX','SY'])
    exdf['TL'] = exdf['TL'].astype('category')
    exdf['SX'] = exdf['SX'].astype('float64')
    exdf['SY'] = exdf['SY'].astype('float64')
    # %%

    es_pred = model.predict(exdf, num_iteration=model.best_iteration) 

    # %%
    exdf.info()
    # %%
    pd.DataFrame(es_pred).describe()


if __name__ == '__main__':
    pytest.main(["-s", "pytesttest.py"])
