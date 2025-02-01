# Model Training 
# Import necessary library and package

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import mysql.connector
from mysql.connector import Error

import os
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pickle

load_dotenv()

try:
    # Step 1: Connect to the database
    conn = mysql.connector.connect(
        host='localhost',
        user='root',
        password=os.getenv('MYSQL_PASS'),
        database='sentiment_analysis_database'
    )

    # Step 2: Fetch all data into a pandas DataFrame
    query = "SELECT id, review_text, sentiment FROM imdb_reviews"
    data = pd.read_sql(query, conn)

    print(f"Loaded {len(data)} records from the database")

except Error as e:
    print(f"Error processing data: {e}")
    conn.rollback()
finally:
    if conn.is_connected():
        conn.close()



# For Model Training we are using Logistic Regression 

# Encode sentiment labels as binary i.e.(1=positive and 0=negative)
data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

# Train/Test/Validation Split

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(data['review_text'],data['sentiment'],test_size=0.2,random_state=42)

# Additional Split from training set for validation 
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Applying TF-IDF on training data
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.fit_transform(X_val)
X_test_tfidf = vectorizer.fit_transform(X_test)



# Training

# Fit the model on the training data using Logistic Regression 
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Monitor basic metrics on the validation set
# For that, Evaluate the model on the validation set 
y_val_pred = model.predict(X_val_tfidf)
accuracy = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)

print("Validation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1_score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")


# Evaluation 

# Evaluate on test set
y_test_pred = model.predict(X_test_tfidf)
test_accuracy = accuracy_score(y_test,y_test_pred)
test_f1 = f1_score(y_test,y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)

print("Test Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"F1_score: {test_f1:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")

# save the trained model using a pickle file 
with open('logistic_regression_tfidf.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

# Save the vectorizer
with open("Vectorizer.pkl", 'wb') as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Model saved as logistic_regression_tfidf.pkl file sucessfully.")
print("Model saved as Vectorizer.pkl file sucessfully.")
