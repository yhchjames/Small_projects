#%%
import pandas as pd
import numpy as np
# %%
data = pd.read_csv("nelson_rules_time_series.csv")
# %%
data['mean'] = data.value.rolling(window = 100).mean()
data['std'] = data.value.rolling(window = 100).std()
data.loc[0:99,'mean'] = data.loc[99,'mean']
data.loc[0:99,'std'] = data.loc[99,'std']


# %%
# rule one
data['r1'] = 0
data.loc[data['value'] >data['mean']+3*data['std'],'r1'] = 1
# only 4 points break r1
#%%
data.r1.sum()

# %%

# rule two
data['larger_mean'] = 0
data.loc[data['value']>data['mean'],'larger_mean'] = 1
# %%
data['r2'] = data.larger_mean.rolling(window = 9).apply(lambda x: x.all())

# %%
# mark position 1-8 as 1
for i in range(len(data)-8):
    if data.loc[i+8,'r2']==1:
        data.loc[i:i+8,'r2'] = 1

#%%
data.r2.sum()
#  692 points

# %%
#rule three
data['r3'] = data.value.rolling(window=6).apply(lambda x: x.is_monotonic or x.is_monotonic_decreasing)

# %%
# position 1-5 as 1
for i in range(len(data)-5):
    if data.loc[i+5,'r3']==1:
        data.loc[i:i+5,'r3'] = 1

#%%
data.r3.sum()
# 46 points

# %%

#rule four
data['sign_for_alt'] = np.sign(data.value.diff())
data.loc[0,'sign_for_alt'] = -1


# %%
data['check_alt'] = data.sign_for_alt.rolling(window = 2).sum()
data.loc[data['check_alt']==0,'check_alt'] =1

# %%
# %%
data['r4'] = data.check_alt.rolling(window = 14).apply(lambda x: (x==1).all())
# %%
for i in range(len(data)-13):
    if data.loc[i+13,'r4']==1:
        data.loc[i:i+13,'r4'] = 1

#%%
data.r4.sum()
# 14 points

# %%
data.to_csv('calculation_for_nrules.csv')

#%%


clist = ['timestamp','r1','r2','r3','r4']
finaldata = data[clist]


# %%
finaldata['rules'] = ''

# %%
finaldata.loc[finaldata['r1']==1,'rules'] = finaldata.loc[finaldata['r1']==1,'rules'].astype(str)+'1;'
finaldata.loc[finaldata['r2']==1,'rules'] = finaldata.loc[finaldata['r2']==1,'rules'].astype(str)+'2;'
finaldata.loc[finaldata['r3']==1,'rules'] = finaldata.loc[finaldata['r3']==1,'rules'].astype(str)+'3;'
finaldata.loc[finaldata['r4']==1,'rules'] = finaldata.loc[finaldata['r4']==1,'rules'].astype(str)+'4'
fclist = ['timestamp','rules']
finaldata = finaldata[fclist]

# %%
finaldata.to_csv('nelson_rules_checking.csv')
# %%
