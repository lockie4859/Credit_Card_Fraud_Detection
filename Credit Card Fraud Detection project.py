#!/usr/bin/env python
# coding: utf-8

# # Import the dependencies
# 

# In[10]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[23]:


#import the dataset

credit_card_data=pd.read_csv("CreditCard.csv")


# In[24]:


#show first 5 rows
credit_card_data.head()


# In[25]:


#show last 5 rows
credit_card_data.tail()


# In[26]:


#check the shape
credit_card_data.shape


# In[27]:


#dataset information
credit_card_data.info()


# In[28]:


#checking the number of missing values in dataset
credit_card_data.isnull().sum()


# In[29]:


#distribution of legit and fraud data collection
credit_card_data['Class'].value_counts()


# # This data set is highly unbalanced
# 
# '0' --> Normal transection
# 
# '1' --> fraud transection

# In[30]:


#seperating the data for analysis
Legit = credit_card_data[credit_card_data.Class == 0]
Fraud = credit_card_data[credit_card_data.Class == 1]  


# In[31]:


print(Legit.shape)
print(Fraud.shape)


# In[32]:


#statistical measure of data
Legit.Amount.describe()


# In[33]:


Fraud.Amount.describe()


# In[34]:


#compare the values of both transection
credit_card_data.groupby('Class').mean()


# # Under sampling
# Build a dataset containing simillar distribution of legit and fraudulent transection
# 
# number of fraudulent transection == 492
# 

# In[35]:


Legit_sample=Legit.sample(n=492)


# # 
# concatanating two datasets

# In[37]:


new_dataset=pd.concat([Legit_sample,Fraud],axis=0)


# In[38]:


new_dataset.head()


# In[49]:


new_dataset['Class'].value_counts()


# In[45]:


new_dataset.groupby('Class').mean()


# splitting data into features and target

# In[51]:


X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']


# In[52]:


print(X)


# In[53]:


print(Y)


# Split the data into training and testing data

# In[54]:


X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# In[56]:


print(X.shape,X_train.shape,X_test.shape)


# Model training
# 
# Logistic Regression

# In[63]:


model = LogisticRegression()


# In[64]:


#training the logistic regression with training data

model.fit(X_train, Y_train)


# Model Evaluation
# 
# Accuracy Score

# In[66]:


#accuracy on training data
X_train_prediction= model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction,Y_train)


# In[67]:


print('Accuracy on training data is ',training_data_accuracy)


# In[68]:


#aacuracy on test data
X_test_prediction= model.predict(X_test)
testing_data_accuracy=accuracy_score(X_test_prediction,Y_test)


# In[69]:


print("Accuracy of testing data is ",testing_data_accuracy)


# This is a Machine Learning credit card fraud detection project in which we have successfully created a model that can detect that the transaction made by the person is Normal or fraudulent. In this project, we learned how to perform exploratory data analysis. And also we have learned how to handle highly unbalanced datasets using sampling. Also learned about Logistic Regression and how to create a Logistic Regression model.

# In[ ]:




