import numpy as np
import pandas as pd
import numpy as np
# import get_data as gd
# import save_data as sd
from sklearn.decomposition import PCA
# 导入数据
# get_path = r'C:\Users\10756\Desktop\data.xlsx'
# save_path = r'C:\Users\10756\Desktop\save_data.xlsx'
data = pd.read_excel(r'C:\Users\10756\Desktop\data1.xlsx')
data.dropna(axis=0, how='any', inplace=True)
# data = data.values.tolist()
data = np.matrix(data)
# 标准化处理
maxium = np.max(data, axis=0)  # 每列最大值
minium = np.min(data, axis=0)  # 每列最小值
data = (data - minium) * 1.0 / (maxium - minium)  # 数据标准化处理
# sd.saveCsv(save_path.format("数据标准化结果.csv"), data)  # 保存数据

# 计算相关统计量
cf = np.cov(data,rowvar=0)  # 计算协方差矩阵
c, d = np.linalg.eig(cf)  # 特征值和特征向量
x = c/np.sum(c)  # 各主成分贡献率，即权重
# sd.saveCsv(save_path.format("协方差矩阵"), cf)
# sd.saveCsv(save_path.format("特征值"), np.transpose(c))
# sd.saveCsv(save_path.format("特征向量"), d)
# sd.saveCsv(save_path.format("各主成分贡献率"), np.transpose(x))
