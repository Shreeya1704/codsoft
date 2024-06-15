import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Assuming we have a dataset movies.csv with columns 'plot' and 'genres'
# For demonstration, let's create a small example dataset
data = {
    'plot': [
        "A young boy discovers he has magical powers and attends a school for wizards.",
        "A group of friends go on an epic adventure to destroy a powerful ring.",
        "A young woman falls in love with a vampire.",
        "A superhero fights to save the world from an alien invasion.",
        "A detective investigates a series of mysterious murders."
    ],
    'genres': [
        ['Fantasy', 'Adventure'],
        ['Fantasy', 'Adventure'],
        ['Romance', 'Fantasy'],
        ['Action', 'Sci-Fi'],
        ['Mystery', 'Thriller']
    ]
}

df = pd.DataFrame(data)

# Preprocessing and splitting the data
X = df['plot']
y = df['genres']

# Convert genres to a binary format
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(y)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF and Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear'), n_jobs=1)),
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict and evaluate the model
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=mlb.classes_))

# Example prediction
new_plot = ["A young woman discovers a secret world and must protect it from an evil force."]
predicted_genre = pipeline.predict(new_plot)
print("\nPredicted Genre:", mlb.inverse_transform(predicted_genre))