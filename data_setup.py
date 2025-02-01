# Import Required Packages

import mysql.connector
from mysql.connector import Error
import os 
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

def create_database():
    try:
        # Establish a connection to the database
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password=os.getenv('MYSQL_PASS')
        )
        cursor = conn.cursor()
        
        # Create database
        cursor.execute("CREATE DATABASE IF NOT EXISTS sentiment_analysis_database")
        cursor.execute("USE sentiment_analysis_database")
        
        # Create imbd_reviews table for storing review_text and sentiment
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS imdb_reviews (
                id INT AUTO_INCREMENT PRIMARY KEY,
                review_text TEXT NOT NULL,
                sentiment VARCHAR(10) NOT NULL
            )
        """)
        
        print("Database and table created successfully")
        conn.commit()
        
    except Error as e:
        print(f"Error creating database: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

# Define load_data function to Load dataset into the database
def load_data():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password=os.getenv('MYSQL_PASS'),
            database='sentiment_analysis_database'
        )
        cursor = conn.cursor()
        
        # Load CSV data
        csv_path = 'IMDB Dataset.csv'
        data = pd.read_csv(csv_path)
        
        # Insert the data into database
        for _, row in data.iterrows():
            cursor.execute(
                "INSERT INTO imdb_reviews (review_text, sentiment) VALUES (%s, %s)",
                (row['review'], row['sentiment'])
            )
        
        conn.commit()
        print(f"Inserted {len(data)} raw reviews")
        
    except Error as e:
        print(f"Error loading data: {e}")
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    create_database()
    load_data()
    

