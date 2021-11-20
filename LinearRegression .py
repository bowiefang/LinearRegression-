#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from IPython.display import Image
import warnings
import sklearn
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


data = pd.read_csv('train.csv')
print(data.shape, data.shape)
data.head()


# In[4]:


# 选择数据集合中的几个重要的特征
data_select = data[['BedroomAbvGr','LotArea','Neighborhood', 'SalePrice']]
data_select = data_select.rename(columns = {'BedroomAbvGr':'room', 'LotArea':'area'})
data_select = data_select.dropna(axis = 0)
for col in np.take(data_select.columns,[0,1,-1]):
    data_select[col] /= data_select[col].max()
data_select.head()


# In[11]:


import sklearn
from sklearn.model_selection  import train_test_split
train, test = train_test_split(data_select.copy(), test_size = 0.9)


# In[14]:


train.describe()


# In[16]:


def linear(features, pars):
    price = np.sum(features*pars[:-1], axis = 1) + pars[-1]
    return price


# In[17]:


# par1 = 0.1, par2 = 0.1
train['predict'] = linear(train[['room', 'area']].values, np.array([0.1, 0.1, 0.0]))
train.head()


# In[18]:


def mean_squared_error(y_pred , y):
    return sum(np.array( y_pred - y ) ** 2)
    
def Cost(df, features, pars):
    df['predict'] = linear(df[features].values, pars)
    cost = mean_squared_error(df.predict, df.SalePrice)/len(df)
    return cost

cost=Cost(train,['room', 'area'], np.array([0.1, 0.1, 0.0]))
print (cost)


# In[19]:


Xs = np.linspace(0, 1, 100)
Ys = np.linspace(0, 1, 100)
Zs = np.zeros([100,100])

Xs,Ys = np.meshgrid(Xs,Ys)
Xs.shape, Ys.shape


# In[24]:


W1=[]
W2=[]
Costs =[]
for i in range (100):
    for j in range(100):
            W1.append(0.01*i)
            W2.append(0.01*j)
            Costs.append(Cost(train,['room', 'area'], np.array([0.01*i, 0.01*j, 0.])))
index = np.array(Costs).argmin()
print (W1[index], W2[index],Costs[index])


# In[26]:


from mpl_toolkits.mplot3d import Axes3D
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.view_init(5,-15)
ax.scatter(W1,W2,Costs,s=10)
ax.scatter(0.58, 0.28, zs = Cost(train,['room', 'area'], np.array([0.58, 0.28, 0.0]) ), s=100,
           color='red')
plt.xlabel('rooms')
plt.ylabel('lotArea')

fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.view_init(5,15)
ax.scatter(W1,W2,Costs,s=10)
ax.scatter(0.58, 0.28, zs = Cost(train,['room', 'area'], np.array([0.58, 0.28, 0.0]) ), s=100,
           color='red')
plt.xlabel('rooms')
plt.ylabel('lotArea')


# In[28]:


# pars 是 w和b的总和
def gradient(train,features, pars):
    Gradient = np.zeros(len(pars))
    for i in range(len(pars)):
        pars_new = pars.copy()
        pars_new[i] += 0.01 
        Gradient[i] = (Cost(train, features, pars_new) - Cost(train,features, pars))/0.01
    return Gradient
gradient(train, ['room', 'area'],[0.2, 0.1, 0])


# In[29]:


def GradientDescent(data, epochs, lr, features, pars):
    Costs = []
    for i in range (epochs):
        grad = gradient(data, features, pars)
        if i%50 == 0:
            Costs.append(Cost(data, features, pars))
        pars -= grad*lr
    print ('w = ', pars)
    return pars, Costs


# In[30]:


pars,Costs = GradientDescent(train, 500, 0.002,['room', 'area'],[0.1, 0.1, 0] )


# In[31]:


cost = Cost(train,['room', 'area'], pars)
cost


# In[32]:


from sklearn.metrics import mean_squared_error
train['predict'] = linear(train[['room', 'area']].values, pars )
print('MSE train: %.3f' % (mean_squared_error(train['SalePrice'], train['predict'])))
train.head()


# In[33]:


cost = Cost(test,['room', 'area'], pars)
cost


# In[34]:


class LinearRegressionGD(object):

    def __init__(self, eta=0.001, n_iter=20):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[:-1] += self.eta * X.T.dot(errors)
            self.w_[-1] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[:-1]) + self.w_[-1]

    def predict(self, X):
        return self.net_input(X)


# In[35]:


data_select = pd.get_dummies(data_select)
train, test = train_test_split(data_select.copy(), test_size = 0.9)


# In[37]:


train_ = train.copy()
train_y = train_.pop('SalePrice')
train_x = train_
test_ = test.copy()
test_y = test_.pop('SalePrice')
test_x = test_


# In[38]:


train_x.head()


# In[39]:


lr = LinearRegressionGD(n_iter = 1000)
lr.fit(train_x.values, train_y.values)


# In[40]:


lr.w_


# In[41]:


from sklearn.metrics import mean_squared_error
train['predict'] = lr.predict(train_x.values)


# In[42]:


print('MSE train: %.3f' % (mean_squared_error(train_y, train['predict'])))
train.head()


# In[43]:


test['predict'] = lr.predict(test_x.values)
print('MSE test: %.3f' % (mean_squared_error(test_y, test['predict'])))
test.head()


# In[ ]:




