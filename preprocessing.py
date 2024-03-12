import pandas as pd 
import plotly.express as px
import numpy as np


df = pd.read_csv('data1.csv')

corr_matrix = df.corr()

# print(corr_matrix)

# 假设corr_matrix是相关系数矩阵
corr_matrix[np.arange(corr_matrix.shape[0])] = np.nan  # 对角线设置为NaN
fig = px.imshow(corr_matrix, color_continuous_scale='Blues')
fig.update_layout(title = "multicollinearity matrix")
fig.show()
