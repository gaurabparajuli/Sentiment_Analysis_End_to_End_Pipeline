# Import necessary libraries and packages

import mysql.connector
from mysql.connector import Error
import pandas as pd
import re
import string
from dotenv import load_dotenv
import os

import matplotlib.pyplot as plt 
from wordcloud import WordCloud 

load_dotenv()

def clean_text(text):
    """Clean and preprocess the text."""
    text = text.lower()
    text = re.sub(r'<br\s*/?>', ' ', text)  # Remove HTML tags
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return ' '.join(text.split())  # Remove extra whitespace

def clean_and_update_data():
    """Clean the data using pandas and update the database."""
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
        df = pd.read_sql(query, conn)

        print(f"Loaded {len(df)} records from the database")

        # Step 3: Clean the data
        # Remove rows with null values
        df = df.dropna(subset=['review_text', 'sentiment'])
        print(f"Removed {len(df) - df.dropna().shape[0]} rows with null values")

        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['review_text', 'sentiment'])
        print(f"Removed {initial_count - len(df)} duplicate rows")

        # Clean the review text
        df['review_text'] = df['review_text'].apply(clean_text)

        # Step 4: Clear the existing table and upload cleaned data
        cursor = conn.cursor()

        # Clear the table
        cursor.execute("DELETE FROM imdb_reviews")
        print("Cleared existing data from the table")

        # Upload cleaned data
        for _, row in df.iterrows():
            cursor.execute(
                "INSERT INTO imdb_reviews (review_text, sentiment) VALUES (%s, %s)",
                (row['review_text'], row['sentiment'])
            )

        conn.commit()
        print(f"Uploaded {len(df)} cleaned records to the database")

    except Error as e:
        print(f"Error processing data: {e}")
        conn.rollback()
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

def performing_eda():
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
        df = pd.read_sql(query, conn)

        print(f"Loaded {len(df)} records from the database")

        # Step 3: Number of reviews per sentiment (distribution)
        # (It means, what is the number of positive and negative reviews by sentiment. Positive sentiment has how many review and Negative Sentiment has how many review.)

        # count the number of reviews per sentiment
        sentiment_counts = df['sentiment'].value_counts()
        # print the results 
        print("Number of reviews per sentiment:")
        print(sentiment_counts)

        # Step 4: Average Review Length for Positive vs Negative Sentiment
        # (It means how long is the review for positive sentiment and how long is the review for negative sentiment in the review_text)
         
        # Calculate the length of each review
        df['review_length'] = df['review_text'].apply(len)

        # Calculate average review length for each sentiment
        avg_length_sentiment = df.groupby('sentiment')['review_length'].mean()

        # print the results
        print("Average review length for each sentiment:")
        print(avg_length_sentiment)
 
        # Step 5: Somme simple plots or word clouds can be included for illustration

        # plots
        # Plotting Barplot for number of reviews per sentiment

        sentiment_counts.plot(kind='bar')
        plt.title("Number of reviews per sentiment")
        plt.xlabel('Sentiment')
        plt.ylabel('Number of Reviews')
        plt.xticks(rotation=0)
        plt.show()


        # Word cloud for Positive and Negative Reviews 
        
        positive_reviews = df[df['sentiment'] == 'positive']['review_text']
        negative_reviews = df[df['sentiment'] == 'negative']['review_text']

        # Join all reviews to create a single string for each sentiment
        positive_text = ' '.join(positive_reviews)
        negative_text = ' '.join(negative_reviews) 


        # Generate word clouds
        positive_word_cloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        negative_word_cloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)

        # Plot the word clouds
        plt.figure(figsize=(10,8))

        # Positive word cloud
        plt.subplot(1,2,1)
        plt.imshow(positive_word_cloud, interpolation='bilinear')
        plt.title('Positive Reviews Word Cloud')
        plt.axis('off')

        # Negative Word Cloud
        plt.subplot(1,2,2)
        plt.imshow(negative_word_cloud, interpolation='bilinear')
        plt.title("Negative Reviews Word Cloud")
        plt.axis('off')

        plt.tight_layout()
        plt.show()


    except Error as e:
        print(f"Error processing data: {e}")
        conn.rollback()
    finally:
        if conn.is_connected():
            conn.close()

if __name__ == "__main__":
    clean_and_update_data()
    performing_eda()
