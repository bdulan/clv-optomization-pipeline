# coding: utf-8

# ***************************Head - Package and Test Problem Size***************************

# In[412]:


# test survival model and seperate optimization by n_seg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as mpl  # plot
import time
import datetime

import statsmodels.api as stmodel
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from xgboost import XGBClassifier
from patsy import dmatrix

from lifelines import AalenAdditiveFitter, CoxPHFitter
from lifelines.utils import k_fold_cross_validation

from mystic.solvers import fmin
from mystic.penalty import quadratic_inequality

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
# %matplotlib notebook


# In[413]:


# Parameter List
n_observation = 100000  # 100k customers
n_attribute = 6  # initial attributes
n_sensitive_seg = 300  # price sensitivity in outcome model

obs_month = 36  # observe max 36 months after CEN
CEN_month = 3  # expiring population month
monthly_discount = 0.0075  # round 9% annual discount for NPV
lead_NPV_m36 = 100  # remaining NPV after 36 months
other_NPV_m36 = 20
mtm_NPV_m36 = 10

p_testA = [-0.5]  # test price decrease to check distribution in Step 5
p_testB = [0.5]  # test price increase to check distribution in Step 5
x_sample_point = 100  # plot objective curve

price_change_lower = -2  # lower bound of price change
price_change_upper = 5  # upper bound of price change
lead_efl2k_lower = 7  # lower bound of lead efl2k
lead_efl2k_upper = 22  # upper bound of lead efl2k
n_expire_seg = lead_efl2k_upper - lead_efl2k_lower + 1  # additional dimension by expire efl2k range
n_expire_seg

# ***************************Step 1: Generate Random Input Attribute and Outcome***************************

# In[414]:


# generate input variables/all random fixed after this step
np.seed = 1
X = np.random.normal(size=(n_observation, n_attribute))  # input attributes/standard normal with negative
input = pd.DataFrame(X)
input['customer_ID'] = range(len(input))
input['lead_efl2k_change'] = input.iloc[:, 0]
input['expire_efl2k'] = input.iloc[:, 1]
input['past_due'] = input.iloc[:, 2]
input['last_bill'] = input.iloc[:, 3]
input['outcome_rand'] = np.random.uniform(size=n_observation)

# price sensitivity category 0 low to n_sensitive_seg high - dwell/AMB/autopay/WSE..
cat_name = np.linspace(1, n_sensitive_seg, num=n_sensitive_seg)
input['sensitive_seg'] = pd.qcut(input.iloc[:, 4], n_sensitive_seg, labels=cat_name)
input['sensitive_seg_num'] = input['sensitive_seg'].cat.codes
input['lead_coeff'] = input['sensitive_seg_num'].apply(lambda x: (x + 1) * (-1) / n_sensitive_seg * 10)
input['reactive_coeff'] = input['sensitive_seg_num'].apply(lambda x: (x + 1) / n_sensitive_seg * 3)

# Normalized usage Jan-Dec/seasonality
input['NRML_KWH1'] = np.random.uniform(0, 1, n_observation) * 2400
input['NRML_KWH2'] = np.random.uniform(0, 1, n_observation) * 2400
input['NRML_KWH3'] = np.random.uniform(0, 1, n_observation) * 2400
input['NRML_KWH4'] = np.random.uniform(0, 1, n_observation) * 2000
input['NRML_KWH5'] = np.random.uniform(0, 1, n_observation) * 2000
input['NRML_KWH6'] = np.random.uniform(0, 1, n_observation) * 2000
input['NRML_KWH7'] = np.random.uniform(0, 1, n_observation) * 3000
input['NRML_KWH8'] = np.random.uniform(0, 1, n_observation) * 3000
input['NRML_KWH9'] = np.random.uniform(0, 1, n_observation) * 3000
input['NRML_KWH10'] = np.random.uniform(0, 1, n_observation) * 2600
input['NRML_KWH11'] = np.random.uniform(0, 1, n_observation) * 2000
input['NRML_KWH12'] = np.random.uniform(0, 1, n_observation) * 2000
input['NRML_KWH'] = input[
    ['NRML_KWH1', 'NRML_KWH2', 'NRML_KWH3', 'NRML_KWH4', 'NRML_KWH5', 'NRML_KWH6', 'NRML_KWH7', 'NRML_KWH8',
     'NRML_KWH9', 'NRML_KWH10', 'NRML_KWH11', 'NRML_KWH12']].sum(axis=1)

# TDSP charge and COGS
input['TDSP'] = pd.qcut(input.iloc[:, 5], 6, labels=['CNP', 'ONC', 'CPL', 'WTU', 'TNP', 'SHY'])
TDSP_cost = pd.DataFrame(
    {'TDSP': ['CNP', 'ONC', 'CPL', 'WTU', 'TNP', 'SHY'], 'TDSP_FIXED': [5.47, 5.25, 9.00, 10.53, 8.65, 10.00],
     'TDSP_VAR': [4.0323, 3.6513, 4.3840, 4.1177, 3.9136, 7.3846], 'COGS': [39.15, 35.12, 39.71, 34.68, 37.67, 35.40]})
input2 = input.merge(TDSP_cost, how='left', on='TDSP')
input2.iloc[0:20, 0:16]


# In[415]:


# actual expire/lead efl2k
def bound(t, L, U):  # general bound
    if t <= L:
        t_new = L
    elif t >= U:
        t_new = U
    else:
        t_new = t
    return t_new


def actual_expire(row):  # actual expire efl2k instead standard normal
    random_EFL2k = 15 + row['expire_efl2k'] * 7  # 7 to 22 cents/KWH
    expire_EFL2k = bound(random_EFL2k, lead_efl2k_lower, lead_efl2k_upper)
    return expire_EFL2k


def actual_lead(row):
    return row['actual_expire_efl2k'] + row['lead_efl2k_change'] * 3


input2['actual_expire_efl2k'] = input2.apply(lambda x: actual_expire(x), axis=1)
# expire offer efl2k category
expire_bin = np.linspace(lead_efl2k_lower - 1, lead_efl2k_upper, num=n_expire_seg + 1)  # include 6<x<=7
expire_cat = np.linspace(1, n_expire_seg, num=n_expire_seg)
input2['expire_seg'] = pd.cut(input2['actual_expire_efl2k'], expire_bin, labels=expire_cat).cat.codes  # 0 to 15
input2['actual_lead_efl2k'] = input2.apply(lambda x: actual_lead(x), axis=1)
# combine sensitivity and expire offer segments
input2['comb_seg'] = input2['expire_seg'].astype(int) * n_sensitive_seg + input2['sensitive_seg_num'].astype(int)
input2.head()
# input2['comb_seg'].unique()


# In[416]:


# add price interaction
price_inter = dmatrix("lead_efl2k_change:sensitive_seg", data=input2, return_type='dataframe')
price_inter2 = price_inter.rename(
    columns=lambda x: x.replace('lead_efl2k_change:sensitive_seg[', 'p').replace('.0]', ''))
model_input = pd.concat([input2, price_inter2], axis=1)
model_input.columns


# In[417]:


# utility function/diff price coefficient for lead and reactive by segment
def lead_offer_utility(row):
    return 0.5 + row['lead_coeff'] * row['lead_efl2k_change'] + 2 * row['expire_efl2k'] - 3 * row['past_due'] + row[
        'last_bill']


def reactive_offer_utility(row):  # other swap to reactive/power tracker/me-too
    return 1 + row['reactive_coeff'] * row['lead_efl2k_change'] + row['expire_efl2k'] - 2 * row['past_due'] + 0.5 * row[
        'last_bill']


def mtm_offer_utility(row):
    return 1 + row['expire_efl2k'] - row['past_due'] + 0.2 * row['last_bill']


def attrition_utility(row):
    return -1 * row['expire_efl2k'] + row['past_due'] - 1 * row['last_bill']


model_input['lead_offer_utility'] = model_input.apply(lambda x: lead_offer_utility(x), axis=1)
model_input['reactive_offer_utility'] = model_input.apply(lambda x: reactive_offer_utility(x), axis=1)
model_input['mtm_offer_utility'] = model_input.apply(lambda x: mtm_offer_utility(x), axis=1)
model_input['attrition_utility'] = model_input.apply(lambda x: attrition_utility(x), axis=1)

# outcome probabilities
model_input['lead_prob'] = np.exp(model_input['lead_offer_utility']) / np.sum(
    [np.exp(model_input['lead_offer_utility']),
     np.exp(model_input['reactive_offer_utility']),
     np.exp(model_input['mtm_offer_utility']),
     np.exp(model_input['attrition_utility'])], axis=0)  # apply to column
model_input['reactive_prob'] = np.exp(model_input['reactive_offer_utility']) / np.sum(
    [np.exp(model_input['lead_offer_utility']),
     np.exp(model_input['reactive_offer_utility']),
     np.exp(model_input['mtm_offer_utility']),
     np.exp(model_input['attrition_utility'])], axis=0)
model_input['mtm_prob'] = np.exp(model_input['mtm_offer_utility']) / np.sum([np.exp(model_input['lead_offer_utility']),
                                                                             np.exp(
                                                                                 model_input['reactive_offer_utility']),
                                                                             np.exp(model_input['mtm_offer_utility']),
                                                                             np.exp(model_input['attrition_utility'])],
                                                                            axis=0)
model_input['att_prob'] = np.exp(model_input['attrition_utility']) / np.sum([np.exp(model_input['lead_offer_utility']),
                                                                             np.exp(
                                                                                 model_input['reactive_offer_utility']),
                                                                             np.exp(model_input['mtm_offer_utility']),
                                                                             np.exp(model_input['attrition_utility'])],
                                                                            axis=0)


# generate actual outcome
def outcome(row):
    if row['outcome_rand'] < row['lead_prob']:
        return 'swap_lead'
    elif row['outcome_rand'] < (row['lead_prob'] + row['reactive_prob']):
        return 'swap_other'
    elif row['outcome_rand'] < (row['lead_prob'] + row['reactive_prob'] + row['mtm_prob']):
        return 'mtm_default'
    else:
        return 'attrition'


model_input['outcome'] = model_input.apply(lambda x: outcome(x), axis=1)
model_input['outcome'].value_counts()


# In[418]:


# observed tenure/status after CEN phase by outcome
def tenure(row):
    if row['outcome'] == 'swap_lead':
        tx = 1 + 2 * row['lead_efl2k_change'] - 1 * row['expire_efl2k'] + 1 * row['past_due'] + 1 * row['last_bill']
        return -np.log(0.3) / (0.25 * np.exp(tx))
    elif row['outcome'] == 'swap_other':
        tx = 1 - 1 * row['expire_efl2k'] + 2 * row['past_due'] + 0.5 * row['last_bill']
        return -np.log(0.3) / (0.10 * np.exp(tx))
    elif row['outcome'] == 'mtm_default':
        tx = 1 - row['expire_efl2k'] + row['past_due'] + 0.2 * row['last_bill']
        return -np.log(0.3) / (0.08 * np.exp(tx))
    else:
        return 0


model_input['tenure'] = model_input.apply(lambda x: tenure(x), axis=1)

# limit to observed period/True inactive/False active at 36 months
model_input['observed_tenure'] = np.round(model_input['tenure'].apply(lambda x: x if x < obs_month else obs_month))
model_input['observed_status'] = model_input['tenure'].apply(lambda x: False if x > obs_month else True)
outcome_class = model_input.groupby(['outcome'])
outcome_mean = outcome_class.mean()
# outcome_mean['observed_tenure']
outcome_mean['observed_status']

# In[419]:


# generate x input for outcome model
x_attribute = model_input.iloc[:, 8:11]
x_inter = model_input.iloc[:, (-13 - n_sensitive_seg):-12]
x_input = pd.concat([x_attribute, x_inter], axis=1)

# generate y outcome label for outcome model
le = LabelEncoder()
le.fit(model_input['outcome'])  # le.inverse_transform(y_outcome)
y_outcome = le.transform(model_input['outcome'])  # 0 attrition/1 mtm_default/2 swap_lead/3 swap_other
x_input.columns

# In[420]:


# generate input for survival model
surv_input = model_input[
    ['customer_ID', 'outcome', 'lead_efl2k_change', 'expire_efl2k', 'past_due', 'last_bill', 'observed_tenure',
     'observed_status']]
swap_lead_sample = surv_input[surv_input.outcome == 'swap_lead'].drop(['customer_ID', 'outcome'], axis=1)
swap_other_sample = surv_input[surv_input.outcome == 'swap_other'].drop(['customer_ID', 'outcome', 'lead_efl2k_change'],
                                                                        axis=1)
mtm_default_sample = surv_input[surv_input.outcome == 'mtm_default'].drop(
    ['customer_ID', 'outcome', 'lead_efl2k_change', ], axis=1)
swap_lead_sample.iloc[0:10, :]

# ***************************Step 2: Model with Lasso Logistic***************************

# In[421]:


lasso = LogisticRegressionCV(Cs=np.logspace(-4, 4, 50), penalty='L1', solver='saga', refit=True)
lasso.fit(x_input, y_outcome)
ls_pred_prob = lasso.predict_proba(x_input)
ls_pred_prob

# In[422]:


lasso.coef_  # price sensitivity match well

# In[423]:


plt.scatter(model_input['lead_prob'], ls_pred_prob[:, 2], color='g', marker='^',
            alpha=0.5)  # c color, alpha transparent 0-1
plt.show()

# In[424]:


plt.scatter(model_input['att_prob'], ls_pred_prob[:, 0], color='g', marker='^',
            alpha=0.5)  # c color, alpha transparent 0-1
plt.show()

# **********************Step 3: Survival Regression by 3 Outcome and Observed 36 Months **********************

# In[425]:


# swap lead survival
cph_surv_lead = CoxPHFitter()
cph_surv_lead.fit(df=swap_lead_sample, duration_col='observed_tenure', event_col='observed_status')
lead_surv_rate = cph_surv_lead.predict_survival_function(
    swap_lead_sample[['lead_efl2k_change', 'expire_efl2k', 'past_due', 'last_bill']]).T
# swap other survival
cph_surv_other = CoxPHFitter()
cph_surv_other.fit(df=swap_other_sample, duration_col='observed_tenure', event_col='observed_status')
other_surv_rate = cph_surv_other.predict_survival_function(
    swap_other_sample[['expire_efl2k', 'past_due', 'last_bill']]).T
# mtm default survival
cph_surv_mtm = CoxPHFitter()
cph_surv_mtm.fit(df=mtm_default_sample, duration_col='observed_tenure', event_col='observed_status')
mtm_surv_rate = cph_surv_mtm.predict_survival_function(mtm_default_sample[['expire_efl2k', 'past_due', 'last_bill']]).T
# combine all outcome
surv_rate = lead_surv_rate.append(other_surv_rate).append(mtm_surv_rate)
surv_output = pd.concat([surv_input, surv_rate], axis=1)
outcome_class = surv_output.groupby(['outcome'])
outcome_mean = outcome_class.mean()
outcome_mean[36.0]  # in line with observed status

# **Step 4: Build Dynamice Probability and Survival Function with EFL2K Price Increase **

# In[426]:


# (P1) outcome/survival model fixed attributes expect dynamic lead_efl2k_change/interaction
cust_fixed_input = model_input[
    ['expire_efl2k', 'past_due', 'last_bill', 'sensitive_seg', 'sensitive_seg_num', 'expire_seg', 'comb_seg',
     'customer_ID', 'actual_expire_efl2k']]
cust_fixed_input.head()

# In[427]:


# (P2) post 36 month margin/NPV input except lead_efl2k_change
# first year 1-12 months/post month after CEN
cust_usage = model_input[
    ['customer_ID', 'NRML_KWH1', 'NRML_KWH2', 'NRML_KWH3', 'NRML_KWH4', 'NRML_KWH5', 'NRML_KWH6', 'NRML_KWH7',
     'NRML_KWH8', 'NRML_KWH9', 'NRML_KWH10', 'NRML_KWH11', 'NRML_KWH12']]
cust_usage2 = cust_usage.rename(columns=lambda x: x.replace('NRML_KWH', ''))
cust_usage_f12 = pd.melt(cust_usage2, id_vars=['customer_ID'], var_name='KWH_Month', value_name='NRML_KWH')
cust_usage_f12['KWH_Month'] = cust_usage_f12['KWH_Month'].astype(int)
cust_usage_f12['Post_Month'] = cust_usage_f12['KWH_Month'].apply(
    lambda x: (x - CEN_month) + 12 if x <= CEN_month                                                              else (
    x - CEN_month))
# second year 13-24 months
cust_usage_s12 = cust_usage_f12.copy()
cust_usage_s12['Post_Month'] = cust_usage_s12['Post_Month'].apply(lambda x: x + 12)
# third year 25-36 months
cust_usage_t12 = cust_usage_f12.copy()
cust_usage_t12['Post_Month'] = cust_usage_t12['Post_Month'].apply(lambda x: x + 24)
cust_usage_p36 = cust_usage_f12.append(cust_usage_s12).append(cust_usage_t12)  # .reset_index()
# expire energe charge/COGS
EC_COGS = model_input[
    ['customer_ID', 'sensitive_seg_num', 'expire_seg', 'comb_seg', 'actual_expire_efl2k', 'COGS', 'TDSP', 'TDSP_FIXED',
     'TDSP_VAR']]
CLV_fixed_input = cust_usage_p36.merge(EC_COGS, how='left', on='customer_ID')


def NPV_discount(row):
    return (1 - monthly_discount) ** row['Post_Month']


CLV_fixed_input['NPV_discount'] = CLV_fixed_input.apply(lambda x: NPV_discount(x), axis=1)
CLV_fixed_input[CLV_fixed_input.customer_ID == 0]


# In[428]:


# (F1) lead efl2k price constraints by customer - depend on price change by comb seg
def lead_efl2k(p, i):
    XL = cust_fixed_input[cust_fixed_input.comb_seg == i].copy()  # local copy
    XL['lead_efl2k_change'] = p[0]
    XL['actual_lead_efl2k'] = XL['actual_expire_efl2k'] + XL['lead_efl2k_change'] * 3
    return XL['actual_lead_efl2k']


p = [0.5]
lead_efl2k(p, 97).head()  # 0 to n_sensitive_seg*n_expire_seg-1


# model_input['comb_seg'].value_counts() #599


# In[429]:


# (F2) customer outcome propensity function by customer - same EFL2k change by comb seg
def cust_outcome_prob(p, i):  # single price/single segment
    XP = cust_fixed_input[cust_fixed_input.comb_seg == i].copy()  # local copy
    XP['lead_efl2k_change'] = p[0]
    price_inter = dmatrix("lead_efl2k_change:sensitive_seg", data=XP, return_type='dataframe')
    price_inter2 = price_inter.rename(
        columns=lambda x: x.replace('lead_efl2k_change:sensitive_seg[', 'p').replace('.0]', ''))
    XP_inter = pd.concat([XP, price_inter2], axis=1)  # append price interation
    XP_inter.drop(XP_inter.columns[[3, 4, 5, 6, 7, 8, 9]], axis=1, inplace=True)  # limit to model variables
    y_pred = lasso.predict_proba(XP_inter)  # 0 attrition/1 mtm_default/2 swap_lead/3 swap_other
    XP['prob_att'] = y_pred[:, 0]
    XP['prob_mtm'] = y_pred[:, 1]
    XP['prob_lead'] = y_pred[:, 2]
    XP['prob_other'] = y_pred[:, 3]
    return XP


p = [0.5]
cust_outcome_prob(p, 97).head()


# In[442]:


# (F3) customer outcome survival function by customer/post month- same EFL2k change by comb seg
def cust_outcome_surv(p, i):  # single price/single segment
    XS = cust_fixed_input[cust_fixed_input.comb_seg == i].copy()
    XS['lead_efl2k_change'] = p[0]
    # lead survival
    lead_surv_rate = cph_surv_lead.predict_survival_function(
        XS[['lead_efl2k_change', 'expire_efl2k', 'past_due', 'last_bill']]).T
    cust_lead_surv = pd.concat([XS['customer_ID'], lead_surv_rate], axis=1)
    cust_lead_surv2 = pd.melt(cust_lead_surv, id_vars=['customer_ID'], var_name='Post_Month',
                              value_name='lead_surv_rate')
    # other survival
    other_surv_rate = cph_surv_other.predict_survival_function(XS[['expire_efl2k', 'past_due', 'last_bill']]).T
    cust_other_surv = pd.concat([XS['customer_ID'], other_surv_rate], axis=1)
    cust_other_surv2 = pd.melt(cust_other_surv, id_vars=['customer_ID'], var_name='Post_Month',
                               value_name='other_surv_rate')
    # mtm survival
    mtm_surv_rate = cph_surv_mtm.predict_survival_function(XS[['expire_efl2k', 'past_due', 'last_bill']]).T
    cust_mtm_surv = pd.concat([XS['customer_ID'], mtm_surv_rate], axis=1)
    cust_mtm_surv2 = pd.melt(cust_mtm_surv, id_vars=['customer_ID'], var_name='Post_Month', value_name='mtm_surv_rate')
    # combine 3 survival
    cust_all_surv = cust_lead_surv2.merge(cust_other_surv2, how='left', on=['customer_ID', 'Post_Month'])
    cust_all_surv2 = cust_all_surv.merge(cust_mtm_surv2, how='left', on=['customer_ID', 'Post_Month'])
    cust_all_surv3 = cust_all_surv2[cust_all_surv2.Post_Month > 0]
    # add seg and lead_efl2k_change
    cust_surv_output = cust_all_surv3.merge(
        XS[['customer_ID', 'sensitive_seg_num', 'expire_seg', 'comb_seg', 'lead_efl2k_change']], how='left',
        on='customer_ID')
    return cust_surv_output


p = [0.5]
# cust_outcome_surv (p, 5).head() #0 to n_seg-1
out = cust_outcome_surv(p, 97)  # 599*36=21,564
out[out.customer_ID == 3477]


# In[431]:


# (F4) customer outcome CLV using F3 by customer - same EFL2k change by comb seg
def cust_outcome_CLV(p, i):
    XC = CLV_fixed_input[CLV_fixed_input.comb_seg == i].copy()
    cust_surv_p36 = cust_outcome_surv(p, i)
    cust_surv_p36.drop(cust_surv_p36.columns[[5, 6, 7]], axis=1, inplace=True)
    cust_p36 = cust_surv_p36.merge(XC, how='left', on=['customer_ID', 'Post_Month'])
    # offer efl2k
    cust_p36['lead_efl2k'] = cust_p36['actual_expire_efl2k'] + cust_p36['lead_efl2k_change'] * 3
    cust_p36['other_efl2k'] = cust_p36['actual_expire_efl2k']
    cust_p36['mtm_efl2k'] = cust_p36['actual_expire_efl2k'] + 0.5  # average 4-6 mil
    # monthly margin
    cust_p36['lead_margin'] = cust_p36['lead_efl2k'] * cust_p36['NRML_KWH'] / 100 - cust_p36['COGS'] * cust_p36[
        'NRML_KWH'] / 1000 - cust_p36['TDSP_VAR'] * cust_p36['NRML_KWH'] / 100 - cust_p36['TDSP_FIXED']
    cust_p36['other_margin'] = cust_p36['other_efl2k'] * cust_p36['NRML_KWH'] / 100 - cust_p36['COGS'] * cust_p36[
        'NRML_KWH'] / 1000 - cust_p36['TDSP_VAR'] * cust_p36['NRML_KWH'] / 100 - cust_p36['TDSP_FIXED']
    cust_p36['mtm_margin'] = cust_p36['mtm_efl2k'] * cust_p36['NRML_KWH'] / 100 - cust_p36['COGS'] * cust_p36[
        'NRML_KWH'] / 1000 - cust_p36['TDSP_VAR'] * cust_p36['NRML_KWH'] / 100 - cust_p36['TDSP_FIXED']
    # monthly margin NPV after discount
    cust_p36['lead_margin_NPV'] = cust_p36['lead_margin'] * cust_p36['NPV_discount']
    cust_p36['other_margin_NPV'] = cust_p36['other_margin'] * cust_p36['NPV_discount']
    cust_p36['mtm_margin_NPV'] = cust_p36['mtm_margin'] * cust_p36['NPV_discount']
    # monthly NPV after attrition/survival
    cust_p36['lead_NPV_surv'] = cust_p36['lead_margin_NPV'] * cust_p36['lead_surv_rate']
    cust_p36['other_NPV_surv'] = cust_p36['other_margin_NPV'] * cust_p36['other_surv_rate']
    cust_p36['mtm_NPV_surv'] = cust_p36['mtm_margin_NPV'] * cust_p36['mtm_surv_rate']
    # sum NPV by customer after discount/survival
    cust_NPV_p36 = cust_p36.groupby(['customer_ID']).sum().reset_index()
    return cust_NPV_p36


p = [0.5]
out = cust_outcome_CLV(p, 97)
out.head()


# np.sum(out['mtm_margin'])/np.sum(out['NRML_KWH'])*1000


# In[432]:


# (F5) customer outcome expected CLV using F2 and F4(F3)
def cust_exp_CLV(p, i):
    cust_out_prob = cust_outcome_prob(p, i)
    cust_out_NPV = cust_outcome_CLV(p, i)
    cust_out_NPV2 = cust_out_NPV[['customer_ID', 'lead_NPV_surv', 'other_NPV_surv', 'mtm_NPV_surv']]
    # combine outcome and NPV post 36 months
    cust_prob_NPV = cust_out_prob.merge(cust_out_NPV2, how='left', on='customer_ID')
    cust_prob_NPV['expected_CLV'] = cust_prob_NPV['prob_att'] * 0 + cust_prob_NPV['prob_mtm'] * (
    cust_prob_NPV['mtm_NPV_surv'] + mtm_NPV_m36) + cust_prob_NPV['prob_other'] * (
    cust_prob_NPV['other_NPV_surv'] + other_NPV_m36) + cust_prob_NPV['prob_lead'] * (
    cust_prob_NPV['lead_NPV_surv'] + lead_NPV_m36)
    return cust_prob_NPV


p = [0.5]
cust_exp_CLV(p, 97).head()


# In[433]:


# [Objective Function] total expected CLV from all customers and outcomes
def objective_CLV(p, i):
    cust_CLV = cust_exp_CLV(p, i)
    total_CLV = -1 * np.sum(cust_CLV['expected_CLV'])
    return total_CLV


p = [0.5]
objective_CLV(p, 97)

# In[434]:


# use p_testA decrease to check each function
a = 17
lead_renewA = cust_outcome_prob(p_testA, a)['prob_lead']  # lead renewal rate
surv_rate = cust_outcome_surv(p_testA, a)
surv_leadA = surv_rate[surv_rate.Post_Month == 36]['lead_surv_rate']  # lead survial 36th month
lead_NPVA = cust_outcome_CLV(p_testA, a)['lead_NPV_surv']  # lead NPV after renew
cust_CLVA = cust_exp_CLV(p_testA, a)['expected_CLV']  # individual expected CLV

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
ax1.hist(lead_renewA)
ax1.set_title("Test A: Lead Renew")
ax2.hist(surv_leadA)
ax2.set_title("Lead Survival 36th")
ax3.hist(lead_NPVA)
ax3.set_title("Lead NPV")
ax4.hist(cust_CLVA)
ax4.set_title("Expected CLV")
f.subplots_adjust(hspace=0.5)

# In[435]:


# use p_testB increase to check each function
lead_renewB = cust_outcome_prob(p_testB, a)['prob_lead']  # lead renewal rate
surv_rate = cust_outcome_surv(p_testB, a)
surv_leadB = surv_rate[surv_rate.Post_Month == 36]['lead_surv_rate']  # lead survial 36th month
lead_NPVB = cust_outcome_CLV(p_testB, a)['lead_NPV_surv']  # lead NPV after renew
cust_CLVB = cust_exp_CLV(p_testB, a)['expected_CLV']  # individual expected CLV

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True)
ax1.hist(lead_renewB)
ax1.set_title("Test B: Lead Renew")
ax2.hist(surv_leadB)
ax2.set_title("Lead Survival 36th")
ax3.hist(lead_NPVB)
ax3.set_title("Lead NPV")
ax4.hist(cust_CLVB)
ax4.set_title("Expected CLV")
f.subplots_adjust(hspace=0.5)

# **********************Step 4: Search Optimial Lead Offer EFL2k Increase by Combination Segment**********************

# In[436]:


begin_sec = time.time()
begin_time = datetime.datetime.fromtimestamp(begin_sec).strftime('%Y-%m-%d %H:%M:%S')
s = 959  # 0 to n_sensitive_seg*n_expire_seg-1(159)


def objective(x):
    objective = objective_CLV(x, s)
    return objective


def penalty_price_change_uplimit(x):  # <= 0 price increase upbound 6
    return x[0] * 3 - price_change_upper


def penalty_price_change_lowlimit(x):  # <= 0 price decrease lowerbound -2
    return -x[0] * 3 + price_change_lower


def penalty_lead_efl2k_uplimit(x):  # lead price 22
    lead_price = lead_efl2k(x, s)  # s=17 feasible 0 to 4 / s=97 feasible -2 to 2 / s = 157 feasible -4 to 0
    return np.max(lead_price) - lead_efl2k_upper


def penalty_lead_efl2k_lowlimit(x):  # lead price 7
    lead_price = lead_efl2k(x, s)
    return -np.min(lead_price) + lead_efl2k_lower


@quadratic_inequality(penalty_price_change_uplimit, k=1e10)
@quadratic_inequality(penalty_price_change_lowlimit, k=1e10)
@quadratic_inequality(penalty_lead_efl2k_uplimit, k=1e20)
@quadratic_inequality(penalty_lead_efl2k_lowlimit, k=1e20)
def penalty(x):
    return 0.0


# feasible x range
expire_efl = cust_fixed_input[cust_fixed_input.comb_seg == s].copy()['actual_expire_efl2k']
feasible_x_lower = max(price_change_lower / 3, (lead_efl2k_lower - np.min(expire_efl)) / 3)  # lead price range
feasible_x_upper = min(price_change_upper / 3, (lead_efl2k_upper - np.max(expire_efl)) / 3)  # price change range

# optimization results
x0 = [min(0.5 / 3, feasible_x_upper)]  # default increase 0.5 or 0 for highest expire efl2k
Nelder_solution = fmin(objective, x0, penalty=penalty, disp=0, retall=1)  # 20-50 interations
print(Nelder_solution[0])
print(f"******************************Optimization Summary******************************")
print(
    f"Segment {s} with expire offer EFL2k from {round(np.min(expire_efl),0)} cents/KWH and sensitivity {s%n_sensitive_seg}")
print(f"Feasible lead/expire offer EFL2k change range from {feasible_x_lower*3} to {feasible_x_upper*3}")
print(f"Optimal lead/expire offer EFL2k change is {Nelder_solution[0][0]*3} instead of default {x0[0]*3} cents/KWH")
print(f"CLV improvement {-objective(Nelder_solution[0])} optimal vs default {-objective(x0)} dollars")
print(f"Price optimization increase CLV by {(objective(Nelder_solution[0])/objective(x0)-1)*100} %")

# running time
end_sec = time.time()
end_time = datetime.datetime.fromtimestamp(end_sec).strftime('%Y-%m-%d %H:%M:%S')
print(f"Process running time from {begin_time} to {end_time}")  # 1-2 minute

# In[437]:


# [A] verify optimal value location
# x test point
x_point = np.linspace(feasible_x_lower * 3, feasible_x_upper * 3, num=x_sample_point)  # price change range
CLV_x = [-objective([i / 3]) for i in x_point]
mpl.plot(x_point, CLV_x)

# feasible/innitial/optimal
mpl.plot(feasible_x_lower * 3, -objective([feasible_x_lower]), 'bo')
mpl.plot(x0[0] * 3, -objective(x0), 'yo')  # default lead/expire offer EFL2k change
mpl.plot(Nelder_solution[0][0] * 3, -objective(Nelder_solution[0]), 'go')  # optimial solution
mpl.plot(feasible_x_upper * 3, -objective([feasible_x_upper]), 'bo')
mpl.title("Maximize CLV within Constraints of Price Change and Lead EFL2K")
mpl.xlabel("Lead/Expire Offer EFL2K Change")
mpl.ylabel("Expected CLV")

# In[438]:


# [B] verify parameter convergence
allvecs = Nelder_solution[-1]
mpl.plot([i[0] * 3 for i in allvecs])
mpl.title("Maximize CLV - Parameter Convergence")
mpl.xlabel("Iterations")
mpl.ylabel("parameter value")
mpl.legend(["p0"])
mpl.show()

# In[439]:


# [C1] verify constraints price change
x_optimal = round(Nelder_solution[0][0] * 3, ndigits=1)
# x_optimal = -5
print(x_optimal)
# price change limit
price_change_range = (price_change_lower, price_change_upper)
price_change_error = f"Optimal lead/expire efl2k change {x_optimal} out of price change limit {price_change_range}"
assert price_change_range[0] <= x_optimal <= price_change_range[1], price_change_error  # error message and jump out

# In[440]:


# [C2] verify constraints lead efl2k range
print(x_optimal)
# lead efl2K limit
lead_efl_range = (lead_efl2k_lower, lead_efl2k_upper)
lead_efl_error = f"Optimial solution with lead efl2k {round(np.min(expire_efl)+x_optimal,1)} to {round(np.max(expire_efl)+x_optimal,1)} out of lead efl2k limit {lead_efl_range}"
assert lead_efl_range[0] <= np.min(expire_efl) + x_optimal <= np.max(expire_efl) + x_optimal <= lead_efl_range[
    1], lead_efl_error


# **********************Step 5: Loop Optimization across Combination Segment**********************

# In[443]:


def optimize_seg_price(k):
    s = k  # comb_seg

    def objective(x):
        objective = objective_CLV(x, s)
        return objective

    def penalty_price_change_uplimit(x):  # <= 0 price increase upbound 6
        return x[0] * 3 - price_change_upper

    def penalty_price_change_lowlimit(x):  # <= 0 price decrease lowerbound -2
        return -x[0] * 3 + price_change_lower

    def penalty_lead_efl2k_uplimit(x):  # lead price 22
        lead_price = lead_efl2k(x, s)  # s=17 feasible 0 to 4 / s=97 feasible -2 to 2 / s = 157 feasible -4 to 0
        return np.max(lead_price) - lead_efl2k_upper

    def penalty_lead_efl2k_lowlimit(x):  # lead price 7
        lead_price = lead_efl2k(x, s)
        return -np.min(lead_price) + lead_efl2k_lower

    @quadratic_inequality(penalty_price_change_uplimit, k=1e10)
    @quadratic_inequality(penalty_price_change_lowlimit, k=1e10)
    @quadratic_inequality(penalty_lead_efl2k_uplimit, k=1e20)
    @quadratic_inequality(penalty_lead_efl2k_lowlimit, k=1e20)
    def penalty(x):
        return 0.0

    # feasible x range
    seg_cust = cust_fixed_input[cust_fixed_input.comb_seg == s].copy()
    expire_efl = seg_cust['actual_expire_efl2k']
    feasible_x_lower = max(price_change_lower / 3, (lead_efl2k_lower - np.min(expire_efl)) / 3)  # lead price range
    feasible_x_upper = min(price_change_upper / 3, (lead_efl2k_upper - np.max(expire_efl)) / 3)  # price change range
    x0 = [min(0.5 / 3, feasible_x_upper)]  # default increase 0.5 or 0 for highest expire efl2k

    # print(f"******************************Optimization by Segment******************************")
    Nelder_solution = fmin(objective, x0, penalty=penalty, disp=0, retall=1)  # 20-50 interations

    # print(Nelder_solution[0])
    # print(f"Segment {s} with expire offer EFL2k from {round(np.min(expire_efl),0)} cents/KWH and sensitivity {s%n_sensitive_seg}")
    # print(f"Feasible lead/expire offer EFL2k change range from {feasible_x_lower*3} to {feasible_x_upper*3}")
    # print(f"Optimal lead/expire offer EFL2k change is {Nelder_solution[0][0]*3} instead of default {x0[0]*3} cents/KWH")
    # print(f"CLV improvement {-objective(Nelder_solution[0])} optimal vs default {-objective(x0)} dollars")
    # print(f"Price optimization increase CLV by {(objective(Nelder_solution[0])/objective(x0)-1)*100} %")

    return Nelder_solution[0][0]


# In[444]:


begin_sec = time.time()
begin_time = datetime.datetime.fromtimestamp(begin_sec).strftime('%Y-%m-%d %H:%M:%S')

k1 = 0
k2 = 4799
seg_point = np.linspace(k1, k2, num=k2 - k1 + 1)  # comb seg range
seg_optimal_efl_change = [optimize_seg_price(k) for k in seg_point]

print(f"******************************Optimization Summary******************************")
print(seg_optimal_efl_change)
# running time
end_sec = time.time()
end_time = datetime.datetime.fromtimestamp(end_sec).strftime('%Y-%m-%d %H:%M:%S')
print(f"Optimization process from {begin_time} to {end_time}")  # 1-2 minute
print(
    f"Total running time is {round(end_sec-begin_sec,0)} seconds for {k2-k1+1} segments, about {round((end_sec-begin_sec)/(k2-k1+1),2)} seconds per segment")

# In[ ]:


optimize_output = pd.dataframe(
    columns=['comb_seg', 'customer_count', 'expire_seg', 'expire_efl2k_lower', 'expire_efl2k_upper',
             'sensitive_seg_num', 'feasible_x_lower', 'feasible_x_upper', 'x_default', 'x_optimal', 'CLV_default',
             'CLV_optimal', 'CLV_increase_pct', 'expected_renew', 'expected_att'])

map_func = {'customer_ID': 'count', 'expire_seg': 'min', 'actual_expire_efl2k': 'min', 'actual_expire_efl2k': 'max',
            'sensitive_seg_num': 'min'}
seg_cust_info = seg_cust.groupby('embark_town').agg(map_func).reset_index()

customer_count = seg_cust.groupby(['comb_seg']).count().reset_index()

cust_fixed_input = model_input[
    ['expire_efl2k', 'past_due', 'last_bill', 'sensitive_seg', 'sensitive_seg_num', 'expire_seg', 'comb_seg',
     'customer_ID', 'actual_expire_efl2k']]

for i in range(5):
    df.loc[i] = [randint(-1, 1) for n in range(3)]

print(df)

