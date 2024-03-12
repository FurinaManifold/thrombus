import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

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
X = df[['Age','Days of hospitalization','Albumin/globulin','Creatinine','Prothrombin time','Fibrinogen','D-dimer']] #'Potassium','Chloride','Fibrin degradation products'
y = df['Presence of thrombus']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize MLP classifier with ReLU activation function
mlp_classifier = MLPClassifier(activation='relu', random_state=0)

# Train the MLP classifier on actual training data
mlp_classifier.fit(X_train, y_train)

# Predict the Test set results
y_pred_mlp = mlp_classifier.predict(X_test)

# Evaluate the model
cm_mlp = confusion_matrix(y_test, y_pred_mlp)
accuracy_mlp = accuracy_score(y_test, y_pred_mlp)

print('Confusion matrix with MLP (ReLU activation):\n', cm_mlp)
print('Model accuracy score with MLP (ReLU activation):', accuracy_mlp)
