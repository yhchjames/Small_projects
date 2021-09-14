#%%
from numpy import testing
from numpy.lib.function_base import median
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lightgbm as lgb
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# %%
data = pd.read_csv("hydropower_time_series.csv")

# Identify transition points


data['medianpow'] = data['power'].rolling(window=20,center=True).median()
data['mapow'] = data['medianpow'].rolling(window=20,center=True).mean()
#%%

next5= data['mapow'].iloc[5:]
next5.reset_index(drop=True, inplace=True)
prev5= data['mapow'].iloc[:-5]
prev5.reset_index(drop=True, inplace=True)

data.reset_index(drop=True, inplace=True)
data['deriv'] = np.abs(next5-prev5)

# %%
data['transition'] = np.where(data['deriv']>0.15,30,0)

# %%
# ax = data.plot(figsize=(500,10))
# fig = ax.get_figure()
# fig.savefig('task1_tran_id.png')
# %%


# %%

#Extract data with sound value

rawdata = data[data['octave_62_5']>0]
# %%
clist = list(rawdata.columns)
fclist = clist[2:7]
fclist.append('transition')
fclist
#%%
rawdata = rawdata[fclist]
# %%
rawdata['transition'] = rawdata['transition']/30
rawdata['transition'] = rawdata['transition'].astype(int)
# %%
rawdata.info()
# %%
#unbalance data, try to discard some data with trans. =0

tran_pos_data = rawdata.loc[rawdata.transition == 1]
tran_non_data = rawdata.loc[rawdata.transition == 0]

discard_data,used_tran_non_data = train_test_split(tran_non_data,test_size = 0.007)

#combined_data is final data set
combined_data = pd.concat([tran_pos_data,used_tran_non_data],ignore_index=True)


# %%
dftrain,dftest = train_test_split(combined_data,test_size=0.1)
# %%
lgb_train = lgb.Dataset(dftrain.drop(['transition'],axis = 1),label=dftrain['transition'])

lgb_valid = lgb.Dataset(dftest.drop(['transition'],axis = 1),label=dftest['transition'],reference=lgb_train)
# %%
params = {
    'boosting_type': 'gbdt',
    'objective':'binary',
    'metric': 'auc',
    'num_leaves': 50,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'max_depth':25

}


results = {}
gbm = lgb.train(params,
                lgb_train,
                num_boost_round= 50,
                valid_sets=(lgb_valid, lgb_train),
                valid_names=('validate','train'),
                early_stopping_rounds = 10,
                evals_result= results)

#%%

y_pred_train = gbm.predict(dftrain.drop('transition',axis = 1), num_iteration=gbm.best_iteration)
y_pred_test = gbm.predict(dftest.drop('transition',axis = 1), num_iteration=gbm.best_iteration)

#all y_pred >0.5 is predicted as 1 
print('train accuracy: {:.5} '.format(accuracy_score(dftrain['transition'],y_pred_train>0.5)))
print('valid accuracy: {:.5} \n'.format(accuracy_score(dftest['transition'],y_pred_test>0.5)))

lgb.plot_metric(results)
lgb.plot_importance(gbm,importance_type = "gain")

