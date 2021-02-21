# -*- coding: utf-8 -*-
# @Time    : 2021/1/29 12:57
# @Author  : JokerTong
# @File    : 单层神经网络的实现.py
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.datasets import load_iris
from torch.optim import SGD
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
#read_excel()用来读取excel文件，记得加文件后缀
data = pd.read_excel(r'C:\Users\10756\Desktop\data.xlsx') 
data.dropna(axis=0, how='any', inplace=True)
y = data['CP二分0，<140,1>=140']  # 目标分类
del data['CP二分0，<140,1>=140']
y = y.values.tolist()
# data =  (data-data.min())/(data.max()-data.min())

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=42)
x_train =  (x_train-x_train.min())/(x_train.max()-x_train.min())
x_test =  (x_test-x_test.min())/(x_test.max()-x_test.min())

pca = PCA(n_components=14)
x_train=pca.fit_transform(x_train)
x_test=pca.fit_transform(x_test)
# matplotlib 中文显示
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# GPU 是否可用
use_cuda = torch.cuda.is_available()
print("use_cuda: ", use_cuda)
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.FloatTensor(x_test)
y_test = torch.LongTensor(y_test)

class Net(torch.nn.Module):
    """
    定义网络
    """

    def __init__(self, n_feature, n_hidden, n_output):
        """
        初始化函数，接受自定义输入特征维数，隐藏层特征维数，输出层特征维数
        """
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(14, 8)  # 一个线性隐藏层
        self.fc2 = torch.nn.Linear(8, 6)
        self.fc3 = torch.nn.Linear(6, 4)
        self.fc4 = torch.nn.Linear(4, 2)
        # self.predict = torch.nn.Linear(n_hidden, n_output)  # 线性输出层

    def forward(self, x):
        """
        前向传播过程
        """
        x = F.relu(self.fc1(x))
        x = F.dropout(x,p=0.2)
        x = F.relu(self.fc2(x))
        x = F.dropout(x,p=0.2)
        x = F.relu(self.fc3(x))
        x = F.dropout(x,p=0.2)        
        pre = self.fc4(x)
        # x = self.predict(x)
        return torch.log_softmax(pre, dim=1)


net = Net(n_feature=10, n_hidden=6, n_output=2)
# 如果GPU可用 训练数据和模型都放到GPU上，注意：数据和网络是否在GPU上要同步
if use_cuda:
    x_train = x_train.cuda()
    y_train = y_train.cuda()
    x_test = x_test.cuda()
    y_test = y_test.cuda()
    net = net.cuda()
print(net)
optimizer = SGD(net.parameters(), lr=0.5)

iter_num = 3000
px, py = [], []

plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
loss_func = nn.MSELoss()
for i in range(iter_num):
    # 数据集传入网络前向计算
    prediction = net(x_train)
    # prediction = torch.FloatTensor(prediction)
    # 计算loss
    loss = F.nll_loss(prediction, y_train)
    # 这里也可用CrossEntropyLoss
    # loss = loss_func(prediction, y)

    # 清除网络状态
    optimizer.zero_grad()

    # loss 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()

    # 打印并记录当前的index 和 loss
    print(i, " loss: ", loss.item())
    px.append(i)
    py.append(loss.item())

    if i % 10 == 0:
        # 动态画出loss走向 结果：loss.png
        plt.cla()
        plt.title('训练过程的loss曲线')
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.plot(px, py, 'r-', lw=1)
        plt.text(0, 0, 'Loss=%.4f' % loss.item(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
    if i == iter_num - 1:
        # 最后一个图像定格.
        plt.show()
        
train_prediction = net(x_train)
count=0       
for i in range(len(y_train)):
    if torch.argmax(train_prediction[i])==y_train[i]:
        count = count+1
pres = count/len(y_train)
print('prediction:'+str(pres))
    
test_prediction = net(x_test)
count=0       
for i in range(len(y_test)):
    if torch.argmax(test_prediction[i])==y_test[i]:
        count = count+1
pres = count/len(y_test)
print('prediction:'+str(pres))