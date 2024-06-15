import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import zipfile
import os

# Step 1: Download and Load Dataset

# URL to download the dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
dataset_path = "smsspamcollection.zip"

# Download the dataset
r = requests.get(url)
with open(dataset_path, 'wb') as f:
    f.write(r.content)

# Unzip the dataset
with zipfile.ZipFile(dataset_path, 'r') as zip_ref:
    zip_ref.extractall()

# Load the dataset into a DataFrame
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])

# Step 2: Preprocessing and Splitting Data

# Map labels to binary values
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Split the data into features and target
X = df['message']
y = df['label']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Helper function to evaluate models
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 3: Model Training and Evaluation

# Naive Bayes
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB()),
])
nb_pipeline.fit(X_train, y_train)
print("Naive Bayes:")
evaluate_model(nb_pipeline, X_test, y_test)

# Logistic Regression
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(max_iter=1000)),
])
lr_pipeline.fit(X_train, y_train)
print("\nLogistic Regression:")
evaluate_model(lr_pipeline, X_test, y_test)

# Support Vector Machine
svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', SVC()),
])
svm_pipeline.fit(X_train, y_train)
print("\nSupport Vector Machine:")
evaluate_model(svm_pipeline, X_test, y_test)