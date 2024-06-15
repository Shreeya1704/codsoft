# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Assume data is your DataFrame containing the dataset
# Example: data = pd.read_parquet('your_dataset.parquet') if the data is in Parquet format
# For demonstration, let's create a synthetic dataset using sklearn's make_classification
from sklearn.datasets import make_classification

# Create a synthetic dataset
data, labels = make_classification(n_samples=10000, n_features=30, n_informative=2, n_redundant=10, random_state=42)
data = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(30)])
data['Class'] = labels

# Data preprocessing
# Separate features and target variable
X = data.drop(columns=['Class'])  # Assuming 'Class' is the target variable
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Standardize the features (only for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train and evaluate Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train_scaled, y_train)
y_pred_log_reg = log_reg.predict(X_test_scaled)
print("Logistic Regression Results:")
print(classification_report(y_test, y_pred_log_reg))
print(confusion_matrix(y_test, y_pred_log_reg))
print("Accuracy:", accuracy_score(y_test, y_pred_log_reg))

# Train and evaluate Decision Tree
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
y_pred_tree = tree.predict(X_test)
print("\nDecision Tree Results:")
print(classification_report(y_test, y_pred_tree))
print(confusion_matrix(y_test, y_pred_tree))
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

# Train and evaluate Random Forest
forest = RandomForestClassifier(random_state=42)
forest.fit(X_train, y_train)
y_pred_forest = forest.predict(X_test)
print("\nRandom Forest Results:")
print(classification_report(y_test, y_pred_forest))
print(confusion_matrix(y_test, y_pred_forest))
print("Accuracy:", accuracy_score(y_test, y_pred_forest))

# Compare models
models = {'Logistic Regression': accuracy_score(y_test, y_pred_log_reg),
          'Decision Tree': accuracy_score(y_test, y_pred_tree),
          'Random Forest': accuracy_score(y_test, y_pred_forest)}

best_model = max(models, key=models.get)
print("\nBest model:", best_model, "with accuracy", models[best_model])