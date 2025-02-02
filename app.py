from flask import Flask, jsonify, request
import pickle
import re
import string 
import os

# Load the trained model and vectorizer 
with open('logistic_regression_tfidf.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Function to preprocess input text
def clean_text(text):
    text = text.lower()
    text = re.sub('<br />', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ''.join(text.split())
    return text 

# Create app route

@app.route('/')
def home():
    return "End to End Sentiment Analysis API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        review_text = data.get('review_text','')

        if not review_text:
            return jsonify({"Error": "No review_text provided"}), 400
        
        # Preprocess input text
        cleaned_text = clean_text(review_text)

        # Transform using TF-IDF
        transformed_text = vectorizer.transform([cleaned_text])

        # Predict Sentiment using trained Logistic Regression model
        prediction = model.predict(transformed_text)[0]

        # Convert numerical prediction to label
        sentiment = 'positive' if prediction == 1 else 'negative'

        return jsonify ({'sentiment_prediction': sentiment})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)