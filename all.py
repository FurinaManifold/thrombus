import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(csv_file_path):
    """Load data from CSV file."""
    try:
        dataframe = pd.read_csv(csv_file_path)
        return dataframe
    except FileNotFoundError:
        print("Error: File not found.")

def preprocess_data(dataframe):
    """Preprocess data."""
    category_counts = dataframe['Main diagnosis'].value_counts()
    categories_to_drop = category_counts[category_counts < 200].index
    dataframe.loc[~dataframe['Main diagnosis'].isin(categories_to_drop), 'Main diagnosis'] = dataframe['Main diagnosis'].mode()[0]

    # Encode labels with value between 0 and n_classes-1
    le = LabelEncoder()
    dataframe['Main diagnosis'] = le.fit_transform(dataframe['Main diagnosis'])

    # Extract features and target variable
    # features = dataframe[['Age', 'BMI', 'Days of hospitalization', 'Albumin/globulin', 'Total protein','Creatinine','Chloride', 'Potassium','Prothrombin time', 'Fibrinogen', 'D-dimer','Fibrin degradation products']]
    features = dataframe[['Age', 'BMI', 'Days of hospitalization', 'Albumin/globulin', 'Total protein','Creatinine','Chloride', 'Potassium','Prothrombin time', 'Fibrinogen', 'D-dimer', 'Age_Albumin', 'BMI_Creatinine']]
    target = dataframe['Presence of thrombus']

    return features, target

def scale_data(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def train_model(model, X_train, y_train):
    """Train model."""
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, accuracy

def plot_confusion_matrix(ax, cm, accuracy, model_name):
    """Plot confusion matrix heatmap."""
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Thrombus', 'Thrombus'], yticklabels=['No Thrombus', 'Thrombus'], ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix - {model_name}\nAccuracy: {accuracy:.2f}')

# Main code
csv_file_path = 'data2.xlsx'
dataframe = pd.read_excel(csv_file_path)
features, target = preprocess_data(dataframe)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=0)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

# Initialize models
logistic_classifier = LogisticRegression(max_iter=100, C=1)
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=0, bootstrap = True)
svm_classifier = SVC(C=1, kernel='linear')
nn_classifier = MLPClassifier(
    activation='relu',  # Use ReLU activation function
    alpha=0.0001,  # Regularization parameter (L2 penalty)
    solver='adam',  # Optimization solver
    learning_rate='adaptive',  # Use a constant learning rate
    learning_rate_init=0.001,  # Initial learning rate
    batch_size='auto',  # Auto determines the batch size based on data size
    max_iter=100,  # Maximum number of iterations
    early_stopping=False,  # Disable early stoppings
    random_state=0  # Random seed for reproducibility
)

classifiers = {'Logistic Regression': logistic_classifier,
               'Random Forest': rf_classifier,
               'Support Vector Machine': svm_classifier,
               'Neural Network': nn_classifier}

# Evaluate and plot confusion matrix for each model
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

for ax, (name, model) in zip(axs.flat, classifiers.items()):
    print(f"Evaluating {name}...")
    model.fit(X_train_scaled, y_train)
    cm, accuracy = evaluate_model(model, X_test_scaled, y_test)
    plot_confusion_matrix(ax, cm, accuracy, name)

plt.tight_layout()
plt.show()

# Plot ROC curve for classifiers that support probability estimates
plt.figure(figsize=(10, 8))

for name, model in classifiers.items():
    print(f"Evaluating {name}...")
    model.fit(X_train_scaled, y_train)
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        auc = roc_auc_score(y_test, y_probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Random classifier
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.show()