
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd


# In[3]:

train_data = pd.read_csv('../../train.csv',sep=',')


# In[4]:

train_data.describe()


# In[5]:

creative_number = train_data.creativeID.unique().shape[0]
user_number = train_data.userID.unique().shape[0]
print("train.csv including userID number is:%s and the creativeID number is: %s."%(user_number,creative_number))
train_data.head()


# In[7]:

app_cate_data = pd.read_csv('../../app_categories.csv',sep=',')
appID_number = np.size(app_cate_data.appID.unique())
appID_appCategory_number = np.size(app_cate_data.appCategory.unique())
print("app_categories.csv including appID number is:%s and categories number is: %s"%(appID_number,appID_appCategory_number))
app_cate_data.head()


# In[8]:

app_cate_data.appCategory.unique()


# In[9]:

ad_data = pd.read_csv('../../ad.csv',sep=',')
appID_number = np.size(ad_data.appID.unique())
creativeID_number = np.size(ad_data.creativeID.unique())
print("ad.csv including appID number is:%s and the creativeID number is: %s."%(appID_number,creativeID_number))
ad_data.head()


# In[10]:

ad_data.appID.unique()


# ## 这里提示我们一个重要信息，就是目前广告推送的app只是有关这50个appID，上面的app_category.csv文件中包含了21万个appID,但是目前和广告相关联的**却只有50个**

# In[11]:

user_app_action_data = pd.read_csv('../../user_app_actions.csv',sep=',')
user_number = np.size(user_app_action_data.userID.unique())
app_number = np.size(user_app_action_data.appID.unique())
print("user_app_action.csv including appID number is:%s and the userID number is: %s."%(app_number,user_number))
ad_data.head()


# 
