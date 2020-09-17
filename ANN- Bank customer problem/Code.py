#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[9]:


df.isnull().sum()


# In[12]:


sns.countplot('Exited',data=df, hue = 'Gender')


# In[13]:


sns.countplot('Exited',data=df, hue = 'HasCrCard')


# In[15]:


df['CreditScore'].nunique()


# In[16]:


df['Surname'].nunique()


# In[18]:


df['Geography'].nunique()


# In[19]:


sns.countplot('Exited',data=df, hue = 'Geography')


# In[20]:


geo = pd.get_dummies(df['Geography'], drop_first = True)


# In[21]:


geo


# In[22]:


df = pd.concat([df.drop('Geography', axis=1), geo], axis = 1)


# In[25]:


df.head()


# In[26]:


df['CustomerId'].nunique()


# In[24]:


gen = pd.get_dummies(df['Gender'], drop_first=True)


# In[28]:


df = pd.concat([df.drop(['Gender', 'RowNumber', 'CustomerId','Surname'], axis = 1 ), gen], axis=1)


# In[29]:


df.head()


# In[31]:


from sklearn.model_selection import train_test_split


# In[41]:


X = df.drop('Exited', axis=1).values


# In[42]:


y = df['Exited'].values


# In[43]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[44]:


from sklearn.preprocessing import MinMaxScaler


# In[45]:


scaler = MinMaxScaler()


# In[46]:


scaler.fit(X_train)


# In[47]:


X_train = scaler.transform(X_train)


# In[48]:


X_test = scaler.transform(X_test)


# In[49]:


X_train.shape


# In[50]:


from tensorflow.keras.models import Sequential


# In[51]:


from tensorflow.keras.layers import Dense, Dropout


# In[68]:


model = Sequential()

model.add(Dense(11, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(6, activation='relu'))
model.add(Dropout(0.1))


model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[69]:


from tensorflow.keras.callbacks import EarlyStopping


# In[70]:


early = EarlyStopping(monitor = 'val_loss',mode='min', patience=10)


# In[71]:


model.fit(x = X_train , y=y_train, epochs = 90 , validation_data = (X_test, y_test), callbacks = [early])


# In[72]:


loss = pd.DataFrame(model.history.history)


# In[73]:


loss


# In[74]:


loss[['loss', 'val_loss']].plot()


# In[75]:


loss[['accuracy', 'val_accuracy']].plot()


# In[82]:


predd = model.predict_classes(X_test)


# In[83]:


predd


# In[84]:


from sklearn.metrics import confusion_matrix, classification_report


# In[85]:


print(confusion_matrix(y_test, predd))
print('/n')
print(classification_report(y_test, predd))


# In[ ]:




