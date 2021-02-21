import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as st
import seaborn as sns
#设置画图属性防止中文乱码
mpl.rcParams['font.sans-serif']=[u'SimHei']
mpl.rcParams['axes.unicode_minus']=False
#引入包
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from sklearn.datasets import load_boston #load data
from sklearn.linear_model import RidgeCV,LassoCV,LinearRegression,ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor

# boston=load_boston()#load data

data1 = pd.read_excel(r'C:\Users\10756\Desktop\data.xlsx') 
data1.dropna(axis=0, how='any', inplace=True)
data1 = (data1-data1.min())/(data1.max()-data1.min())
y = data1['CP二分0，<140,1>=140']  # 目标分类
del data1['CP二分0，<140,1>=140']
# y = y.numpy()
# x = data1.numpy()
x = data1
# y = y.values.tolist()

# x =  (data-data.min())/(data1.max()-data1.min())

# x=boston.data #
# y=boston.target #label

# print("特征的列名")
# print(boston.feature_names)
print("样本数据量:%d,特征个数：%d"%x.shape)
# 
# x=pd.DataFrame(boston.data,columns=boston.feature_names)
x.head()
sns.distplot(tuple(y),kde=False,fit=st.norm)#标签分布可视化

x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.2,random_state=28)
ss= StandardScaler()
x_train=ss.fit_transform(x_train)
x_test=ss.fit_transform(x_test)
#先做标准化再拆分，不然每一段数据的标准会有不同
##bulid model
#model's name
names=['LinerRegression','Ridge','Lasso',
       'Random Forrset','GBDT','Support Vector Regression',
       'ElasticNet','XgBoost']

#define model
#cv 交叉验证思想
models=[LinearRegression(),
        RidgeCV(alphas=(0.001,0.1,1),cv=3),
        LassoCV(alphas=(0.001,0.1,1),cv=5),
        RandomForestRegressor(n_estimators=10), #回归效果好一点
        GradientBoostingRegressor(n_estimators=30), #回归效果好一点
        SVR(),
        ElasticNet(alpha=0.001,max_iter=10000),
        XGBRegressor()] #回归效果好一点
#输出所有回归模型的R2评分

#定义R2评分函数
def R2(model,x_train,x_test,y_train,y_test):
    model_fitted = model.fit(x_train,y_train) #模型训练
    y_pred = model_fitted.predict(x_test) #标签预测
    score =r2_score(y_test,y_pred) #r2得分
    return score

#遍历所有模型进行评分
for name,model in zip(names,models):
    score = R2(model,x_train,x_test,y_train,y_test)
    print("{}:{:.6f},{:.4f}".format(name,score.mean(),score.std()))        

    
#模型构建
"""
'kernel':核函数
'C':SVR的正则化因子
'gamma':'rbf','poly'and 'sigmoid'核函数的系数，影响模型性能
"""
parameters={
    'kernel':['lonear','rbf'],
    'C':[0.1,0.5,0.9,1,5],
    'gamma':[0.001,0.01,0.1,1]
    
    }

model = GridSearchCV(XGBRegressor(), param_grid=parameters,cv=3)
model.fit(x_train,y_train)

#获取最优参数
print("最优参数列表:",model.best_params_)
print("最优模型:",model.best_estimator_)
print("最优R2值:",model.best_score_)

#可视化
In_x_test =range(len(x_test))
y_predict = model.predict(x_test)
plt.figure(figsize=(16,8),facecolor='w')
plt.plot(In_x_test,y_test,'r-',lw=2,label=u'真实值')
plt.plot(In_x_test,y_predict,'g-',lw=3,label=u'SVR算法估计值，$R^2$=%.3f'%(model.best_score_))
plt.legend(loc='upper left')
plt.grid(True)
plt.title("波士顿房价预测（SVM）")
plt.xlim(0,101)
plt.show()

