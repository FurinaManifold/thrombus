from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
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

from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel

svm_classifier = SVC()


# Initialize LassoCV with cross-validation
lasso_cv = LassoCV(cv=5)

# Fit LassoCV on training data
lasso_cv.fit(X_train, y_train)

# Get the best regularization coefficient (lambda value)
best_lambda = lasso_cv.alpha_

# Select features using Lasso regularization
feature_selector = SelectFromModel(lasso_cv, threshold='median')
X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_test_selected = feature_selector.transform(X_test)

# Train your preferred model (e.g., SVM) using the selected features
svm_classifier.fit(X_train_selected, y_train)

# Predict the Test set results
y_pred_svm_selected = svm_classifier.predict(X_test_selected)

# Evaluate the model with selected features
print('Best regularization coefficient (lambda):', best_lambda)
print('Classification report with selected features:')
print(classification_report(y_test, y_pred_svm_selected))
