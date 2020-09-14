#!/usr/bin/env python
# coding: utf-8

# # LINEAR REGRESSION
# PRICING AN INSURANCE POLICY

# In[2]:


# LOADING PACKAGES
# Setting up the working directory
import os
os.chdir("C:/Users/jai/Desktop/PythonCaseStudy")
# Working with dataframes
import pandas as pd
# Mathematical operations
import numpy as np
# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
# Train and test 
from sklearn.model_selection import train_test_split
# Linear Modelling packages
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm


# In[5]:


#DATASET
insurance=pd.read_csv("InsurancePolicy.csv")
print(insurance.head())
print(insurance.info())


# In[26]:


# Numerical summary of the dataset
insurance.describe()


# In[48]:


# Categorical summary
print(insurance['gender'].value_counts())
print(insurance['region'].value_counts())
print(insurance['smoker'].value_counts())
print(insurance['children'].value_counts())


# In[49]:


# Variable Selection - Categorical & Count Variables
sns.boxplot(x='gender',y='charges',data=insurance)
plt.show()
sns.boxplot(x='region',y='charges',data=insurance)
plt.show()
sns.boxplot(x='smoker',y='charges',data=insurance)
plt.show()
sns.boxplot(x='children',y='charges',data=insurance)
plt.show()


# In[52]:


#CORRELATION
ins=insurance.drop(columns=['charges','children'],axis=1)
corr_dt=ins.select_dtypes(exclude=[object])
corr_dt.corr()


# In[56]:


# train & test split and getting dummies
x1=insurance.drop(columns=['charges'],axis=1)
x1=pd.get_dummies(x1,drop_first=True) 
y1=insurance.filter(['charges'],axis=1)
x_train, x_test, y_train, y_test = train_test_split(x1,y1,test_size=0.3,random_state=3)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[57]:


x_train2 = sm.add_constant(x_train)
model_lin1 = sm.OLS(y_train, x_train2)
results1=model_lin1.fit()
print(results1.summary())


# In[58]:


# Predicting 
x_test2=sm.add_constant(x_test)
pred_lin1_test =results1.predict(x_test2)
print(pred_lin1_test)


# In[15]:


residuals=y_train.iloc[:,0]-pred_lin1_train

# Residual plot
sns.regplot(x=pred_lin1_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")


# Plotting the variable price
## For orginal data
prices = pd.DataFrame({"1. Before":y1.iloc[:,0], "2. After":np.log(y1.iloc[:,0])})
prices.hist()

# Plotting the variable price
## Train set
prices = pd.DataFrame({"1. Before":y_train.iloc[:,0], "2. After":np.log(y_train.iloc[:,0])})
prices.hist()


# In[ ]:





# In[16]:


# Transforming prices to log -- Dependent variable on log scale
y2=np.log(y1)
y_train_log,y_test_log=train_test_split(y2, test_size=0.3, random_state = 3)


# In[23]:


x_train2 = sm.add_constant(x_train)
model_lin2 = sm.OLS(y_train_log, x_train2)
results2=model_lin2.fit()
print(results2.summary())

# Predicting model on test set 
x_test=sm.add_constant(x_test)
ins_predictions_lin2_test = results2.predict(x_test)

def rmse_log(test_y,predicted_y):
    # To get non-log scale values
    t1=np.exp(test_y) 
    t2=np.exp(predicted_y)
    rmse_test=np.sqrt(mean_squared_error(t1,t2))
    
    #for base rmse
    base_pred = np.repeat(np.mean(t1), len(t1))
    rmse_base = np.sqrt(mean_squared_error(t1, base_pred))
    values={'RMSE-test from model':rmse_test,'Base RMSE':rmse_base}
    return values

# Model evaluation on predicted and test 
rmse_log(y_test_log,ins_predictions_lin2_test)

ins_predictions_lin2_train = results2.predict(x_train2)
residuals=y_train_log.iloc[:,0]-ins_predictions_lin2_train

# Residual plot
sns.regplot(x=ins_predictions_lin2_train,y=residuals)
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title('Residual plot')

# QQ plot
sm.qqplot(residuals)
plt.title("Normal Q-Q Plot")


# In[18]:


sns.lmplot(x=)


# In[ ]:




