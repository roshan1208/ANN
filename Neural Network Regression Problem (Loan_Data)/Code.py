#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[29]:


data_info = pd.read_csv('../DATA/lending_club_info.csv',index_col='LoanStatNew')


# In[30]:


print(data_info.loc['revol_util']['Description'])


# In[31]:


def feat_info(col_name):
    print(data_info.loc[col_name]['Description'])


# In[32]:


feat_info('mort_acc')


# In[ ]:





# In[26]:


df = pd.read_csv('lending_club_loan_two.csv')


# In[27]:


df.info()


# In[35]:


sns.set_style('darkgrid')
sns.countplot('loan_status', data=df)


# In[39]:


sns.distplot(df['loan_amnt'])


# In[45]:


plt.figure(figsize=(12,4))
sns.distplot(df['loan_amnt'],kde=False,bins=40)
plt.xlim(0,45000)


# In[46]:


df.corr()


# In[49]:


plt.figure(figsize=(14,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')


# In[50]:


feat_info('installment')


# In[52]:


plt.figure(figsize=(14,8))
sns.scatterplot('installment', 'loan_amnt', data=df)


# In[54]:


sns.boxplot( 'loan_status', 'loan_amnt', data=df)


# In[56]:


df.groupby('loan_status').describe()


# In[57]:


df['grade'].unique()


# In[59]:


df.columns


# In[69]:


sorted(df['sub_grade'].unique())


# In[73]:


sns.countplot('grade', data=df, hue='loan_status')


# In[74]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' )


# In[75]:


plt.figure(figsize=(12,4))
subgrade_order = sorted(df['sub_grade'].unique())
sns.countplot(x='sub_grade',data=df,order = subgrade_order,palette='coolwarm' ,hue='loan_status')


# In[76]:


f_and_g = df[(df['grade']=='G') | (df['grade']=='F')]

plt.figure(figsize=(12,4))
subgrade_order = sorted(f_and_g['sub_grade'].unique())
sns.countplot(x='sub_grade',data=f_and_g,order = subgrade_order,hue='loan_status')


# In[77]:


df['loan_status'].unique()


# In[78]:


df['loan_repaid'] = df['loan_status'].map({'Fully Paid':1,'Charged Off':0})


# In[79]:


df[['loan_repaid','loan_status']]


# In[80]:


df.corr()['loan_repaid'].sort_values().drop('loan_repaid').plot(kind='bar')


# In[81]:


df.isnull().sum()


# In[82]:


100* df.isnull().sum()/len(df)


# In[83]:


feat_info('emp_title')
print('\n')
feat_info('emp_length')


# In[84]:


df['emp_title'].nunique()


# In[85]:


df['emp_title'].value_counts()


# In[86]:


df = df.drop('emp_title',axis=1)


# In[87]:


sorted(df['emp_length'].dropna().unique())


# In[88]:


emp_length_order = [ '< 1 year',
                      '1 year',
                     '2 years',
                     '3 years',
                     '4 years',
                     '5 years',
                     '6 years',
                     '7 years',
                     '8 years',
                     '9 years',
                     '10+ years']


# In[90]:


plt.figure(figsize=(12,4))

sns.countplot(x='emp_length',data=df,order=emp_length_order)


# In[91]:


plt.figure(figsize=(12,4))
sns.countplot(x='emp_length',data=df,order=emp_length_order,hue='loan_status')


# In[92]:


emp_co = df[df['loan_status']=="Charged Off"].groupby("emp_length").count()['loan_status']


# In[93]:


emp_fp = df[df['loan_status']=="Fully Paid"].groupby("emp_length").count()['loan_status']


# In[94]:


emp_len = emp_co/emp_fp


# In[95]:


emp_len


# In[96]:


emp_len.plot(kind='bar')


# In[97]:


df = df.drop('emp_length',axis=1)


# In[98]:


df.isnull().sum()


# In[101]:


df['title'].head(10)


# In[102]:


df = df.drop('title',axis=1)


# In[103]:


feat_info('mort_acc')


# In[104]:


df['mort_acc'].value_counts()


# In[105]:


print("Correlation with the mort_acc column")
df.corr()['mort_acc'].sort_values()


# In[106]:


print("Mean of mort_acc column per total_acc")
df.groupby('total_acc').mean()['mort_acc']


# In[107]:


total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# In[108]:


total_acc_avg[2.0]


# In[109]:


def fill_mort_acc(total_acc,mort_acc):
    '''
    Accepts the total_acc and mort_acc values for the row.
    Checks if the mort_acc is NaN , if so, it returns the avg mort_acc value
    for the corresponding total_acc value for that row.
    
    total_acc_avg here should be a Series or dictionary containing the mapping of the
    groupby averages of mort_acc per total_acc values.
    '''
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


# In[110]:


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)


# In[111]:


df.isnull().sum()


# In[112]:


df = df.dropna()


# In[113]:


df.isnull().sum()


# In[114]:


df.info()


# In[115]:


df.select_dtypes(['object']).columns


# In[116]:


df['term'].value_counts()


# In[117]:


df['term'] = df['term'].apply(lambda term: int(term[:3]))


# In[118]:


df = df.drop('grade',axis=1)


# In[119]:


subgrade_dummies = pd.get_dummies(df['sub_grade'],drop_first=True)


# In[120]:


subgrade_dummies.head()


# In[121]:


df = pd.concat([df.drop('sub_grade',axis=1),subgrade_dummies],axis=1)


# In[122]:


df.columns


# In[123]:


df.select_dtypes(['object']).columns


# In[124]:


dummies = pd.get_dummies(df[['verification_status', 'application_type','initial_list_status','purpose' ]],drop_first=True)
df = df.drop(['verification_status', 'application_type','initial_list_status','purpose'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[125]:


df.columns


# In[126]:


df['home_ownership'].value_counts()


# In[127]:


df['home_ownership']=df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')

dummies = pd.get_dummies(df['home_ownership'],drop_first=True)
df = df.drop('home_ownership',axis=1)
df = pd.concat([df,dummies],axis=1)


# In[129]:


df['zip_code'] = df['address'].apply(lambda address:address[-5:])


# In[130]:


df['zip_code'].unique()


# In[131]:


dummies = pd.get_dummies(df['zip_code'],drop_first=True)
df = df.drop(['zip_code','address'],axis=1)
df = pd.concat([df,dummies],axis=1)


# In[132]:


df.select_dtypes(['object']).columns


# In[133]:


df = df.drop('issue_d',axis=1)


# In[134]:


df['earliest_cr_year'] = df['earliest_cr_line'].apply(lambda date:int(date[-4:]))
df = df.drop('earliest_cr_line',axis=1)


# In[135]:


df.select_dtypes(['object']).columns


# In[136]:


from sklearn.model_selection import train_test_split


# In[140]:


df = df.drop('loan_status',axis=1)


# In[144]:


df = df.sample(frac=0.1,random_state=101)


# In[146]:


len(df)


# In[142]:


X = df.drop('loan_repaid',axis=1).values
y = df['loan_repaid'].values


# In[143]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)


# In[148]:


X_train.shape


# In[150]:


X_test.shape


# In[151]:


from sklearn.preprocessing import MinMaxScaler


# In[152]:


scaler = MinMaxScaler()


# In[153]:


X_train = scaler.fit_transform(X_train)


# In[154]:


X_test = scaler.transform(X_test)


# In[155]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation,Dropout
from tensorflow.keras.constraints import max_norm


# In[188]:


model = Sequential()

model.add(Dense(78,  activation='relu'))
model.add(Dropout(0.1))

# hidden layer
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.1))

# hidden layer
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.1))

# output layer
model.add(Dense(units=1,activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam')


# In[189]:


from tensorflow.keras.callbacks import EarlyStopping


# In[190]:


earlystop = EarlyStopping(monitor = 'val_loss', mode='min', verbose = 1, patience = 20)


# In[191]:


model.fit(x=X_train, 
          y=y_train, 
          epochs=100,
          batch_size=256,
          validation_data=(X_test, y_test), 
          callbacks=[earlystop]
          )


# In[192]:


from tensorflow.keras.models import load_model


# In[193]:


model.save('My_portfolio_project_model.h5')  


# In[194]:


losses = pd.DataFrame(model.history.history)


# In[195]:


losses.plot()


# In[196]:


from sklearn.metrics import classification_report,confusion_matrix


# In[197]:


predictions = model.predict_classes(X_test)


# In[198]:


print(classification_report(y_test,predictions))


# In[199]:


confusion_matrix(y_test,predictions)


# In[200]:


import random
random.seed(101)
random_ind = random.randint(0,len(df))

new_customer = df.drop('loan_repaid',axis=1).iloc[random_ind]
new_customer


# In[201]:


model.predict_classes(new_customer.values.reshape(1,78))


# In[202]:


df.iloc[random_ind]['loan_repaid']


# In[ ]:




