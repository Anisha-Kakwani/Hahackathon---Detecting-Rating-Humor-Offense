#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd, numpy as np, re, time
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# In[6]:


data = pd.read_csv('train.csv')

#data = dataset[['text', 'is_humor']]

print(data.head())
print('-------------------------------------------------------------------------')
print(data.isnull().any(axis = 0))


# In[7]:


# Relacing special symbols and digits in headline column
# re stands for Regular Expression
data['text'] = data['text'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))


# In[8]:


print(data.head())
print('-------------------------------------------------------------------------')
print(data.isnull().any(axis = 0))

# getting features and labels
features = data['text']
labels = data['is_humor']


# In[9]:


# Stemming our data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())

features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
print(features)


# In[10]:


# vectorizing the data with maximum of 5000 features
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 3223)
features = list(features)
features = tv.fit_transform(features).toarray()
print(features)


# In[12]:


# getting training and testing data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.2, random_state = 0)


# In[15]:


# model 1:-
# Using linear support vector classifier
lsvc = LinearSVC()
# training the model
lsvc.fit(features_train, labels_train)
#getting the score of train and test data
print(lsvc.score(features_train, labels_train)) # 90.93
print(lsvc.score(features_test, labels_test))   # 83.75
# model 2:-
# Using Gaussuan Naive Bayes
# gnb = GaussianNB()
# gnb.fit(features_train, labels_train)
# print(gnb.score(features_train, labels_train))  # 78.86
# print(gnb.score(features_test, labels_test))    # 73.80
# # model 3:-
# # Logistic Regression
# lr = LogisticRegression()
# lr.fit(features_train, labels_train)
# print(lr.score(features_train, labels_train))   # 88.16
# print(lr.score(features_test, labels_test))     # 83.08
# # model 4:-
# # Random Forest Classifier
#rfc = RandomForestClassifier(n_estimators = 10, random_state = 0)
#rfc.fit(features_train, labels_train)
#print(rfc.score(features_train, labels_train))  # 98.82
#print(rfc.score(features_test, labels_test))    # 79.71


# In[16]:


eval_data = pd.read_csv('public_dev.csv')
print(eval_data)


# In[17]:


eval_df = eval_data['text'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))
print(eval_df)
print("------------------------------------------------------------------")
eval_df_features = eval_df.apply(lambda x: x.split())

eval_df_features = eval_df_features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))
print(eval_df_features)


# In[18]:


eval_df_features = list(eval_df_features)
tv1 = TfidfVectorizer(max_features = 3223)

eval_df_features = tv1.fit_transform(eval_df_features).toarray()
print(eval_df_features)
print(len(eval_df_features))


# In[19]:


# eval_df_features=tv1.transform(eval_df_features)
#classifier.predict(X_test)
humor_predict = lsvc.predict(eval_df_features)


# In[20]:


print(humor_predict)


# In[ ]:




