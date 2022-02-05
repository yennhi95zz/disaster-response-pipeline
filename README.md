# Project: Disaster Response Pipeline

## Project Overview
I gained and improved my data engineering abilities in the Udacity - Data Scientist Nanodegree Program, which has increased my opportunities and potential as a data scientist. In this project, I'll use these talents to analyse disaster data from [Figure Eight](https://www.figure-eight.com/) in order to create a model for a disaster-messaging API.

A data collection in the Project Workspace contains genuine messages sent during disaster occurrences. I'll build a machine learning pipeline to classify these occurrences and deliver the notifications to the relevant disaster relief organisation.

A web app will be included in the project, which will allow an emergency worker to enter a new message and receive classification results in numerous categories. The data will be visualised on the web app as well.

A few screenshots of the web app are shown below.
![image](https://user-images.githubusercontent.com/88694623/152629914-8ad0e597-f8dc-49d1-a01a-a6f41db4f74d.png)
![image](https://user-images.githubusercontent.com/88694623/152629918-a14224c9-8edc-4d9c-85d2-b3105f42098b.png)

## File structure
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv  # data to process
|- process_data.py
|- InsertDatabaseName.db   # database to save clean data to

- models
|- train_classifier.py
|- classifier.pkl  # saved model 

- README.md
```
## Project Components
There are three components in this project.

### 1. ETL Pipeline
In a Python script, `process_data.py`, write a data cleaning pipeline that:

- Loads the `messages` and `categories` datasets
- Merges the two datasets
- Cleans the data
- Stores it in a SQLite database
### 2. ML Pipeline
In a Python script, `train_classifier.py`, write a machine learning pipeline that:

- Loads data from the SQLite database
- Splits the dataset into training and test sets
- Builds a text processing and machine learning pipeline
- Trains and tunes a model using GridSearchCV
- Outputs results on the test set
- Exports the final model as a pickle file
### 3. Flask Web App
Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
Your web app should now be running if there were no errors.

Now, open another Terminal Window.

Type
```
env|grep WORK
```
You'll see output that looks something like this:
![image](https://user-images.githubusercontent.com/88694623/152629785-d5f1fef4-2011-4466-9163-c3960c8dd98b.png)
In a new web browser window, type in the following:
```
https://SPACEID-3001.SPACEDOMAIN
```
In this example, that would be: "https://viewa7a4999b-3001.udacity-student-workspaces.com/" (Don't follow this link now, this is just an example.)

Your SPACEID might be different.

You should be able to see the web app. The number 3001 represents the port where your web app will show up. Make sure that the 3001 is part of the web address you type in.






