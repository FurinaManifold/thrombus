from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Specify the path to your CSV file
csv_file_path = 'data1.csv'

# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Drop categories with less than 200 data points
category_counts = df['Main diagnosis'].value_counts()
categories_to_drop = category_counts[category_counts < 200].index
df = df[~df['Main diagnosis'].isin(categories_to_drop)]

# Encode labels with value between 0 and n_classes-1
le = preprocessing.LabelEncoder()
df['Main diagnosis'] = le.fit_transform(df['Main diagnosis'])

# Extract features and target variable
X = df[['Age', 'Days of hospitalization', 'Albumin/globulin', 'D-dimer']]
y = df['Presence of thrombus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the SVM classifier
svm_classifier = SVC()

# Define the parameter grid to search
param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Penalty parameter C
    'gamma': ['scale', 'auto'],  # Kernel coefficient for 'rbf' kernel
    'kernel': ['linear', 'rbf']  # Kernel type
}

# Initialize GridSearchCV for SVM with 5-fold cross-validation
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5)

# Train the SVM classifier using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters found by GridSearchCV
best_params = grid_search.best_params_

# Initialize the SVM classifier with the best parameters
best_svm_classifier = SVC(**best_params)

# Train the SVM classifier with the best parameters on actual training data
best_svm_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_svm = best_svm_classifier.predict(X_test)

# Evaluate the model
print("Best parameters found by GridSearchCV:", best_params)
print("Classification report:")
print(classification_report(y_test, y_pred_svm))
