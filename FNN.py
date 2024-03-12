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


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(80, activation = 'tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(80, activation = 'tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(80, activation = 'tanh'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(80, activation = 'tanh'),
    tf.keras.layers.Dense(2, activation = 'softmax')
])

model.compile(optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ['sparse_categorical_accuracy']
)

'''
checkpoint_save_path = "./checkpoint/fashion_CNN.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):
    print('----------load the model----------')
    model.load_weights(checkpoint_save_path)
'''

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

'''
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint_save_path, save_weights_only = True, save_best_only = True)

'''
history = model.fit(X_train, y_train, batch_size = 32, epochs = 100, validation_data = (X_test, y_test), validation_freq = 1) #, callbacks = [cp_callback])

model.summary()

acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

Y_pred = tf.argmax(model.predict(X_test), axis = 1)

# 计算混淆矩阵
cm = confusion_matrix(y_test, Y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Thrombus', 'Thrombus'], yticklabels=['No Thrombus', 'Thrombus'])

# 显示图形
plt.show()

plt.subplot(1,2,1)
plt.plot(acc, label = 'Training Accuracy')
plt.plot(val_acc, label = 'Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.subplot(1,2,2)
plt.plot(loss, label = 'Training Loss')
plt.plot(val_loss, label = 'Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()