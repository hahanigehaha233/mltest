
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from pylab import mpl
from sklearn import cross_validation
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题


# In[2]:


figsize=(12,5)
regex_a = "Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass.*|AgePclass"
data_train = pd.read_csv('titanic_data/train.csv')
scaler = preprocessing.StandardScaler()


# In[3]:


# The model of feture Age
def set_Age_Gap(df):
    df.loc[(df.Age.isnull()),'Age'] = df.Age.mean()
    return df

def set_Age_Discrete(df):
    tmp_age = df.Age
    factor = pd.cut(tmp_age, [0,15,70,100],labels = ['Age_young','Age_adult','Age_old'])
    dummies_age_Discrete = pd.get_dummies(factor)
    return dummies_age_Discrete


# In[4]:


# The model of feture Cabin 
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


# In[5]:


# The model of Fare
def set_Fare_scale(df):
    return scaler.fit_transform(df['Fare'].values.reshape(-1,1))


# In[6]:


def set_Feature(data_train):
    data_train = set_Age_Gap(data_train)

    data_train = set_Cabin_type(data_train)

    dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix= 'Cabin')

    dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix= 'Embarked')

    dummies_Sex = pd.get_dummies(data_train['Sex'], prefix= 'Sex')

    dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix= 'Pclass')

    b = set_Age_Discrete(data_train)

    df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,b], axis=1)

    df['Fare_scaled'] = set_Fare_scale(data_train)

    # New feature AgePclass from 0.77 to 0.775
    df['AgePclass'] = df['Age']*df['Sex_female']
    df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked','Fare','Age'], axis=1, inplace=True)
    
    return df


# In[7]:


def get_X_y(df):
    train_np = df.filter(regex=regex_a).values
    
    X = train_np[:,1:]
    
    y = train_np[:,0]
    
    return X,y


# In[8]:



df = set_Feature(data_train=data_train)
# 用正则取出我们要的属性值
X,y = get_X_y(df)

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=0.5,penalty='l1',tol=1e-6)
clf.fit(X, y)

clf


# In[10]:


# 对cross validation数据进行预测
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_titanic = Pipeline([
                        ('clf',LogisticRegression(C=1000.0,penalty='l2',tol=1e-4))])
pipe_titanic.fit(X,y)


# In[14]:


data_test = pd.read_csv("Titanic_data/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
df_test = set_Feature(data_test)
test = df_test.filter(regex=regex_a)
predictions = pipe_titanic.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("Titanic_data/pip.csv", index=False)
test.info()

