#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing Libraries and reading the dataset

import numpy as np # linear algebra
import pandas as pd # data processing,
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("C:/Users/harul/Documents/ASSIGNMENT/PA PROJECT DATASET.csv")
df.head(10)


# In[2]:


# information about the Dataset
df.info()


# In[3]:


# summary on the  price attribute
df.describe()


# In[4]:


# printing the column and rows
df.shape


# In[5]:


# displaying the Null Values
df.isnull().sum()


# In[6]:


#drop the nullvalues
df.dropna(inplace=True)


# In[7]:


# displaying the changed null values
df.isnull().sum()


# In[8]:


# displaying the datatypes
df.dtypes


# In[9]:



def change_into_datetime(col):
    df[col]=pd.to_datetime(df[col])
df.columns    


# In[10]:


# changing the object attribute to datetime datatypes
for i in ['Date_of_Journey','Dep_Time', 'Arrival_Time']:
    change_into_datetime(i)
df.dtypes    


# In[11]:


# Sepearting the  Date_of_Journey to Date and Month 
df['journey_date']=df['Date_of_Journey'].dt.day
df['journey_month']=df['Date_of_Journey'].dt.month
df.head(10)


# In[12]:


# droping the Date_of_Journey Attribute
df.drop('Date_of_Journey', axis=1, inplace=True)


# In[13]:


# function for extracting hour and minutes
def extract_hour(data,col):
    data[col+'_hour']=data[col].dt.hour
    
def extract_min(data,col):
    data[col+'_min']=data[col].dt.minute
    

def drop_col(data,col):
    data.drop(col,axis=1,inplace=True)


# In[14]:


# extracting hours from Dep_Time
extract_hour(df,'Dep_Time')

#extracting minutes
extract_min(df,'Dep_Time')

#drop the column
drop_col(df,'Dep_Time')


# In[15]:


# extracting hours from Arrival_Time
extract_hour(df,'Arrival_Time')

#extracting min
extract_min(df,'Arrival_Time')

#drop the column
drop_col(df,'Arrival_Time')


# In[16]:


df.head(10)


# In[17]:


duration=list(df['Duration'])
for i in range(len(duration)):
    if len(duration[i].split(' '))==2:
        pass
    else:
        if 'h' in duration[i]: # Check if duration contains only hour
             duration[i]=duration[i] + ' 0m' # Adds 0 minute
        else:
             duration[i]='0h '+ duration[i]


# In[18]:


df['Duration']=duration
df.head()


# In[19]:


# spliting the Duration into Hours and Seconds
def hour(x):
    return x.split(' ')[0][0:-1]

def minutes(x):
    return x.split(' ')[1][0:-1]
df['dur_hour']=df['Duration'].apply(hour)
df['dur_min']=df['Duration'].apply(minutes)
df.head(10)


# In[20]:


# droping the Duration 
drop_col(df,'Duration')

# Displaying the datatypes
df.dtypes


# In[21]:


# converting the object datatypes to integer datatype
df['dur_hour'] = df['dur_hour'].astype(int)
df['dur_min'] = df['dur_min'].astype(int)
df.dtypes


# In[22]:


# displaying the Categorical Variables
column=[column for column in df.columns if df[column].dtype=='object']
column


# In[23]:


# displaying the Numerical Variables
continuous_col =[column for column in df.columns if df[column].dtype!='object']
continuous_col


# In[24]:


# displaying the Categorical Variables
categorical = df[column]
categorical.head()


# In[25]:


# Categories in Airline Attribute
categorical['Airline'].value_counts()


# In[26]:


# import the Seaborn Packages and Ploting the Airline with Price Attribute
import seaborn as sns
plt.figure(figsize=(15,8))
sns.boxplot(x='Airline',y='Price',data=df.sort_values('Price',ascending=False))


# In[27]:


# Ploting the Total_Stops with Price Attribute
plt.figure(figsize=(15,8))
sns.boxplot(x='Total_Stops',y='Price',data=df.sort_values('Price',ascending=False))


# In[28]:


# Using the One-Hot Encoding the Airline Datatypes
Airline=pd.get_dummies(categorical['Airline'],drop_first=True)
Airline.head()


# In[29]:


# Categories in Source Attribute
categorical['Source'].value_counts()


# In[30]:


# Ploting the Source with Price Attribute
plt.figure(figsize=(15,15))
sns.catplot(x='Source',y='Price',data=df.sort_values('Price',ascending=False),kind='boxen')


# In[31]:


# Categories in Destination Attribute
plt.figure(figsize=(15,15))
sns.catplot(x='Destination',y='Price',data=df.sort_values('Price',ascending=False),kind='boxen')


# In[32]:


# Using the One-Hot Encoding the Source Datatypes
source=pd.get_dummies(categorical['Source'],drop_first=True)
source.head()


# In[33]:


# Categories in Destination Attribute
categorical['Destination'].value_counts()


# In[34]:


# Ploting the Destination with Price Attribute
plt.figure(figsize=(15,8))
sns.boxplot(x='Destination',y='Price',data=df.sort_values('Price',ascending=False))


# In[35]:


# Using the One-Hot Encoding the Destination Datatypes
destination=pd.get_dummies(categorical['Destination'],drop_first=True)
destination.head()


# In[36]:


# Displaying the Categories in Categorial Attribute
for i in categorical.columns:
    print('{} has total {} categories'.format(i,len(categorical[i].value_counts())))


# In[37]:


# Ploting the arrival_time_hour with price attribute
df.plot.hexbin(x='Arrival_Time_hour',y='Price',gridsize=15)


# In[38]:


# Categories in Total_Stops Attribute
categorical['Total_Stops'].unique()


# In[39]:


# Concatinating the all Categorical variable
data=pd.concat([categorical,Airline,source,destination,df[continuous_col]],axis=1)
data.head()


# In[40]:


# Identifying the outlier
def plot(data,col):
    fig,(ax1,ax2)=plt.subplots(2,1)
    sns.distplot(data[col],ax=ax1)
    sns.boxplot(data[col],ax=ax2)
plot(data,'Price')


# In[41]:


# Resolving the outliers
data['Price']=np.where(data['Price']>=40000,data['Price'].median(),data['Price'])
plot(data,'Price')


# In[42]:


# encoding Total stops
dict={'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
categorical['Total_Stops']=categorical['Total_Stops'].map(dict)


# In[43]:


# changing the Categorical to numerical datatypes using label encoding
categorical['Total_Stops']


# In[44]:


# droping the Categorical Variable attribute
drop_col(categorical,'Source')
drop_col(categorical,'Destination')
drop_col(categorical,'Airline')


# In[46]:


data=pd.concat([categorical,Airline,source,destination,df[continuous_col]],axis=1)
data.head()


# In[47]:


# Displaying the all Columns 
pd.set_option('display.max_columns',33)
data.head()


# In[48]:


# assigning the X and Y values
X=data.drop('Price',axis=1)
y=df['Price']


# In[49]:


# importing the mutual_information from sklearn packages
# Mutual information (MI) [1] between two random variables is a non-negative value, 
# which measures the dependency between the variables. It is equal to zero 
# if and only if two random variables are independent, and higher values mean higher dependency.


from sklearn.feature_selection import mutual_info_classif
mutual_info_classif(X,y)


# In[50]:


imp = pd.DataFrame(mutual_info_classif(X,y),index=X.columns)
imp


# In[51]:


imp.columns=['importance']
imp.sort_values(by='importance',ascending=False)


# In[52]:


# importing the train_test from sklearn model selection
# Spliting the test and train data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=123)


# In[53]:


# importing the r2_score, mean_absolute_error, mean_squared_error

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
def predict(ml_model):
    print('Model is: {}'.format(ml_model))
    model= ml_model.fit(X_train,y_train)
    print("Training score: {}".format(model.score(X_train,y_train)))
    predictions = model.predict(X_test)
    print("Predictions are: {}".format(predictions))
    print('\n')
    r2score=r2_score(y_test,predictions) 
    print("r2 score is: {}".format(r2score))
          
    print('MAE:{}'.format(mean_absolute_error(y_test,predictions)))
    print('MSE:{}'.format(mean_squared_error(y_test,predictions)))
    print('RMSE:{}'.format(np.sqrt(mean_squared_error(y_test,predictions))))
     
    sns.distplot(y_test-predictions) 


# In[54]:


# importing the all types regression

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor


# In[55]:


predict(RandomForestRegressor())


# In[56]:


predict(LogisticRegression())


# In[57]:


predict(KNeighborsRegressor())


# In[58]:


predict(DecisionTreeRegressor())


# In[59]:


from sklearn.svm import SVR
predict(SVR())


# In[60]:


predict(GradientBoostingRegressor())


# In[ ]:




