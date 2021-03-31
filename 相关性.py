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
from torch.optim import Adam
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import numpy as np
import math
#read_excel()用来读取excel文件，记得加文件后缀
data = pd.read_excel(r'C:\Users\10756\Desktop\data1.xlsx') 
data.dropna(axis=0, how='any', inplace=True)
data2=data.corr(method='spearman')
# print(data2)
data3=abs(data2['Cp   mg/l']).sort_values(ascending=False)
print(data3)

