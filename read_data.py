import pandas as pd
#read_excel()用来读取excel文件，记得加文件后缀
data = pd.read_excel(r'C:\Users\10756\Desktop\data.xlsx') 
data.dropna(axis=0, how='any', inplace=True)
data2 =  (data-data.min())/(data.max()-data.min())

print('显示表格的属性:',data.shape)   #打印显示表格的属性，几行几列
print('显示表格的列名:',data.columns) #打印显示表格有哪些列名
