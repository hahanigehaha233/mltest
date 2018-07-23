
# coding: utf-8

# In[3]:


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


# In[ ]:


figsize=(12,5)
regex_a = "Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_*|Embarked_.*|Sex_.*|Pclass.*|AgePclass"
data_train = pd.read_csv('titanic_data/train.csv')
scaler = preprocessing.StandardScaler()


# In[2]:


# The model of feture Age
def set_Age_Gap(df):
    df.loc[(df.Age.isnull()),'Age'] = df.Age.mean()
    return df

def set_Age_Discrete(df):
    tmp_age = df.Age
    factor = pd.cut(tmp_age, [0,15,70,100],labels = ['Age_young','Age_adult','Age_old'])
    dummies_age_Discrete = pd.get_dummies(factor)
    return dummies_age_Discrete


# In[ ]:


# The model of feture Cabin 
def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df


# In[4]:


# The model of Fare
def set_Fare_scale(df):
    return scaler.fit_transform(df['Fare'].values.reshape(-1,1))


# In[15]:


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


# In[7]:




# 用正则取出我们要的属性值
train_df = df.filter(regex=regex_a)
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=0.5,penalty='l1',tol=1e-6)
clf.fit(X, y)

clf


# In[16]:


#简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex=regex_a)
X = all_data.values[:,1:]
y = all_data.values[:,0]
print(cross_validation.cross_val_score(clf, X, y, cv=5))
all_data


# In[18]:


# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train, split_cv = cross_validation.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex=regex_a)
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.values[:,1:], train_df.values[:,0])

# 对cross validation数据进行预测
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_titanic = Pipeline([
                        ('clf',LogisticRegression(C=1000.0,penalty='l2',tol=1e-4))])
pipe_titanic.fit(X,y)
pipe_titanic.fit(train_df.values[:,1:], train_df.values[:,0])

cv_df = split_cv.filter(regex=regex_a)
predictions = pipe_titanic.predict(cv_df.values[:,1:])

origin_data_train = pd.read_csv("Titanic_data/train.csv")
bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.values[:,0]]['PassengerId'].values)]
bad_cases.info()


# In[46]:


pd.DataFrame({"columns":list(train_df.columns)[1:], "coef":list(clf.coef_.T)})


# In[19]:


data_test = pd.read_csv("Titanic_data/test.csv")
data_test.loc[ (data_test.Fare.isnull()), 'Fare' ] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]
data_test = set_Age_Gap(data_test)

data_test = set_Cabin_type(data_test)

dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix= 'Cabin')

dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix= 'Embarked')

dummies_Sex = pd.get_dummies(data_test['Sex'], prefix= 'Sex')

dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix= 'Pclass')

b = set_Age_Discrete(data_test)

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass,b], axis=1)

df_test['AgePclass'] = df_test['Age']*df_test['Sex_female']
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Fare_scaled'] = set_Fare_scale(data_test)
test = df_test.filter(regex=regex_a)
predictions = pipe_titanic.predict(test)
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values, 'Survived':predictions.astype(np.int32)})
result.to_csv("Titanic_data/pip.csv", index=False)
test.info()

