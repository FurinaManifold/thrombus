import tensorflow as tf 
import os
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import pandas as pd
import numpy as np

# Specify the path to your CSV file
csv_file_path = 'data2.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Drop categories with less than 200 data points
category_counts = df['Main diagnosis'].value_counts()
categories_to_drop = category_counts[category_counts < 200].index
df = df[~df['Main diagnosis'].isin(categories_to_drop)]

# Encode labels with value between 0 and n_classes-1
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
df['Main diagnosis'] = le.fit_transform(df['Main diagnosis'])

# Extract features and target variable
# X = df[['Age','Days of hospitalization','Albumin/globulin','Creatinine','Prothrombin time','Fibrinogen','D-dimer', 'Age_Albumin', 'BMI_Creatinine']] #'Potassium','Chloride','Fibrin degradation products'
X = df[['Age','Albumin/globulin','Creatinine','Prothrombin time','Fibrinogen','D-dimer']]
y = df['Presence of thrombus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# 假设 model 是您想要作为基模型的分类器
# 这里我们使用决策树作为示例
base_model = DecisionTreeClassifier()

# 创建 BaggingClassifier 实例
# n_estimators 是基模型的数量，random_state 是随机数种子
# features_fraction 是每个基模型使用的特征比例
# bootstrap 是布尔值，如果为 True 则使用 bootstrapping
bagging_model = RandomForestClassifier(
    estimator=base_model,
    n_estimators=100,  # 例如，使用 100 个基模型
    random_state=42,
    features_fraction=1.0,  # 使用所有特征
    bootstrap=True  # 使用 bootstrapping 来构建每个基模型
)

# 训练 Bagging 模型
bagging_model.fit(X_train, y_train)

# 预测测试集
Y_pred = bagging_model.predict(X_test)

# 评估模型性能
# 这里使用 accuracy_score 作为评估指标
accuracy = accuracy_score(y_test, Y_pred)
print(f"Model Accuracy: {accuracy}")