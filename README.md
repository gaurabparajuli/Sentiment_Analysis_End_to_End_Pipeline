# End-to-End Sentiment Analysis Pipeline

## Project Setup 
### Create a virtual environment
Creating a virtual environment to do a project is a good way to do the work. for that, create a folder and open that folder using a ide. In my case i used vs code. Then open a new terminal and type
python -m venv name_of_your_virtual_environment 

After your virtual environment is created activate it. 
For that in terminal type

cd name_of_your_virtual_environment
Scripts\activate

now, your venv is activated. 

### Installing Dependencies
Install the required dependencies from requirements.txt file 
e.g. pip install reuirements.txt 

Now your all required dependencies are installed. Now you are good to go ahead. 

### Store your MYSQL user credential in environment variable if you haven't 
For that go to environment variable and select "User Variables for YOUR_USERNAME"
Click on NEW button and type 
Variable Name: MYSQL_PASS 
Variable Value: your_mysql_user_password

### Create a .env file to keep your MYSQL database credential
so, in my case i created a .env file in same folder where we are doing our coding. In vs code i created a new file called .env and inside the file i wrote the following: 

MYSQL_PASS = your_mysql_user_password

### Data Acquisition
I downloaded the dataset from Kaggle. link:https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

About Dataset: 

IMDB dataset having 50K movie reviews for natural language processing or Text analytics.
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training and 25,000 for testing. So, predict the number of positive and negative reviews using either classification or deep learning algorithms.[source:Kaggle]

### Run Instruction:
Now, let's see how to run the scripts. 
At first run, 
data_setup.py

In this step we did the following thing:
1. Create Database Function
a. Establish a connection to MySQL server
b. create database
c. create table
d. Close the connection
2. Load Data (csv file) to Mysql
a. Establish a connection to MySQL Server
b. Load the dataset using pandas
c. Insert the dataset into database
d. Close the connection

At second run,
data_cleaning_and_exploration.py

In this step we did the following thing:
1. Clean and Update data by fetching MySQL data into pandas data frame and reupload it into database. 
2. Performing EDA like:
a. Number of Reviews per Sentiment
b. Average Review Length for Positive vs Negative Sentiment
c. Bar PLot and word clouds

At third run,
train_model.py

In this step we did the following thing:
1. Connect to the database
2. Fetch all data into a pandas dataframe
3. Encode Sentiment lanels as binary
4. Train/Test Split
5. Additional split from training set for Validation
6. Applying TF-IDF on training data 
7. Training the model using Logistic Regression
8. Evaluate and Print Metrics on the Validation set.
9. Evaluate and Print Metrics on the Test set
10. Save the trained model using a pickle file
11. Save the Vectorizer using a pickle file. 

and at last run,
app.py

In this step we did the following thing:
1. Load the trained model and vectorizer
2. Initialize Flask app
3. Function to preprocess the input text
4. Create a route using POST method
5. Define a function to predict the sentiment of given input text i.e.(review_text)
a. Get the json file as input
b. Preprocess the input text (clean the text)
c. Transform using TF-IDF 
d. Predict Sentiment using trained Logistic Regression Model
e. Convert Numerical Prediction to Label



