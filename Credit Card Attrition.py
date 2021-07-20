#!/usr/bin/env python
# coding: utf-8

# In[233]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[394]:


data = pd.read_csv('Banking_CreditCardAttrition.csv',sep=';')


# In[395]:


data


# In[236]:


data.info()


# In[237]:


data.isnull().sum()


# In[238]:


data.describe()


# ## Data Preprocessing 

# In[239]:


data.drop('CLIENTNUM',axis = 1,inplace =True)
data   


# In[240]:


data['Attrition_Flag'].unique()


# In[241]:


#customer_age
data.Customer_Age.count()


# In[242]:


data.Customer_Age.describe()


# In[243]:


data.Customer_Age.isnull().sum()


# In[244]:


data['Customer_Age'].fillna(data['Customer_Age'].mean(),inplace = True)
#data.Customer_Age.isnull().sum()


# In[245]:


#Gender
data['Gender'].value_counts()
sns.countplot(x ='Attrition_Flag',hue ='Gender',  data = data)


# In[246]:


#Dependent count
data.Dependent_count.value_counts()


# In[247]:


#Education_level
data.Education_Level.value_counts()


# In[248]:


#As we can see there is unknown data so we can replace it with graduate one.

data["Education_Level"].replace({"Unknown":np.NaN}, inplace=True)
data.Education_Level = data.Education_Level.fillna('Graduate')


# In[249]:


fig = plt.figure(figsize=(10,5))
sns.countplot(x = 'Education_Level',hue = 'Attrition_Flag',data =data)


# In[250]:


#Marital status
data.Marital_Status.value_counts()


# In[251]:


fig = plt.figure(figsize=(10,5))
sns.countplot(x = 'Marital_Status',hue = 'Attrition_Flag',data =data)


# In[252]:


#Income_Category
data.Income_Category.value_counts()


# In[253]:


fig = plt.figure(figsize=(10,5))
sns.countplot(x = 'Income_Category',hue = 'Attrition_Flag',data =data)


# In[254]:


#Card_Category
data.Card_Category.value_counts()


# In[255]:


fig = plt.figure(figsize=(10,5))
sns.countplot(x = 'Card_Category',hue = 'Attrition_Flag',data =data)


# In[256]:


#Credit_Limit
data.Credit_Limit.describe()


# In[291]:


data['Credit_Limit'].fillna(data['Credit_Limit'].mean(),inplace = True)


# In[257]:


#Adding all Transaction amount in 6 months
data['Six_Months_Trans_Amt'] = data.Trans_Amt_Oct12 + data.Trans_Amt_Nov12 + data.Trans_Amt_Dec12 + data.Trans_Amt_Jan13 + data.Trans_Amt_Feb13 + data.Trans_Amt_Mar13


# In[258]:


data['Six_Months_Trans_Count'] = data.Trans_Count_Oct12 + data.Trans_Count_Nov12 + data.Trans_Count_Dec12 + data.Trans_Count_Jan13 + data.Trans_Count_Feb13 + data.Trans_Count_Mar13


# In[259]:


data.drop(['Trans_Amt_Oct12','Trans_Amt_Nov12','Trans_Amt_Dec12','Trans_Amt_Jan13','Trans_Amt_Feb13','Trans_Amt_Mar13'
              ,'Trans_Count_Oct12','Trans_Count_Nov12','Trans_Count_Dec12','Trans_Count_Jan13',
              'Trans_Count_Feb13','Trans_Count_Mar13'], axis = 1,inplace = True)


# In[260]:


data.head(5)


# ## Detecting Outlier

# In[327]:


#using IQR for 

def Outliers_IQR(dt):
    q1, q3 = np.percentile(dt, [25, 75])
    IQR_x = q3 - q1
    lower_limit = q1 - (IQR_x * 1.5)
    upper_limit = q3 + (IQR_x * 1.5)
    sns.boxplot(x = dt.index, data=dt,color=".25")


# In[328]:


Outliers_IQR(data['Customer_Age'])
min_threshold, max_threshold = data.Customer_Age.quantile([0.05,0.99])
min_threshold, max_threshold


# In[329]:


data.Dependent_count.describe()


# In[330]:


Outliers_IQR(data['Dependent_count'])
min_threshold, max_threshold = data.Dependent_count.quantile([0.05,0.99])
min_threshold, max_threshold


# In[331]:


Outliers_IQR(data['Months_on_book'])
min_threshold, max_threshold = data.Months_on_book.quantile([0.05,0.99])
min_threshold, max_threshold


# In[332]:


Outliers_IQR(data['Total_Relationship_Count'])
min_threshold, max_threshold = data.Total_Relationship_Count.quantile([0.05,0.99])
min_threshold, max_threshold


# In[333]:


Outliers_IQR(data['Months_Inactive_12_mon'])
min_threshold, max_threshold = data.Months_Inactive_12_mon.quantile([0.05,0.99])
min_threshold, max_threshold


# In[334]:


Outliers_IQR(data['Credit_Limit'])
min_threshold, max_threshold = data.Credit_Limit.quantile([0.05,0.99])
min_threshold, max_threshold


# In[335]:


Outliers_IQR(data['Contacts_Count_12_mon'])
min_threshold, max_threshold = data.Contacts_Count_12_mon.quantile([0.05,0.99])
min_threshold, max_threshold


# In[336]:


Outliers_IQR(data['Six_Months_Trans_Amt'])
min_threshold, max_threshold = data.Six_Months_Trans_Amt.quantile([0.05,0.99])
min_threshold, max_threshold


# In[337]:


Outliers_IQR(data['Six_Months_Trans_Amt'])
min_threshold, max_threshold = data.Six_Months_Trans_Amt.quantile([0.05,0.99])
min_threshold, max_threshold


# In[338]:


Outliers_IQR(data['Total_Revolving_Bal'])
min_thresold, max_thresold = data.Total_Revolving_Bal.quantile([0.05,0.99])
min_thresold, max_thresold


# In[378]:


def filling_Outliers(yy):
    minm = yy.quantile(0.05)
    maxm = yy.quantile(0.99)
    yy.fillna(maxm)
    yy.fillna(minm)


# In[379]:


filling_Outliers(data.Months_on_book)
filling_Outliers(data.Months_Inactive_12_mon)
filling_Outliers(data.Contacts_Count_12_mon)
filling_Outliers(data.Six_Months_Trans_Amt)
filling_Outliers(data.Six_Months_Trans_Count)
filling_Outliers(data.Credit_Limit)


# In[341]:


data.head(5)


# In[380]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def Cal_vif(data):
    vif = pd.DataFrame()
    vif['features'] = data.columns
    vif['VIF_value'] = [variance_inflation_factor(data.values, i) for i in range(data.shape[1])]
    return vif


# In[381]:


temp_data = data[['Customer_Age','Dependent_count','Months_on_book','Total_Relationship_Count','Months_Inactive_12_mon',
                'Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Six_Months_Trans_Amt','Six_Months_Trans_Count']]
temp_data.head()


# In[382]:


temp_data = temp_data.drop(['Months_on_book','Customer_Age','Six_Months_Trans_Count'], axis =1)
temp_data.head()


# In[383]:


data.isnull().sum()


# In[384]:


Calculate_VIF (New_data)


# In[385]:


t_data = data[['Attrition_Flag','Dependent_count','Total_Relationship_Count','Months_Inactive_12_mon','Contacts_Count_12_mon','Credit_Limit','Total_Revolving_Bal','Six_Months_Trans_Amt']]
t_data.head(3)


# ## Splitting Data

# In[386]:


x = t_data.drop(['Attrition_Flag'], axis =1)
y = t_data['Attrition_Flag']


# ## Logistic Regression

# In[414]:


from  sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# In[415]:


X_train, X_test, Y_train, Y_test = train_test_split(x,y, test_size=0.4, random_state=42)


# In[416]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)


# In[417]:


pred = model_LR.predict(X_test)


# In[418]:


pred = LR.predict(X_test)
print(classification_report(Y_test,prediction))


# In[ ]:





# In[ ]:




