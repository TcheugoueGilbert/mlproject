from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)  # Create a Flask application instance

app=application

## Route for a home page

@app.route('/')  # Define the route for the home page
def index():    # Render the index.html template when the home page is accessed
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])  # Define the route for the prediction page, allowing both GET and POST methods
def predict_datapoint():  # Handle the prediction request
    if request.method=='GET':  # If the request method is GET, render the home.html template
        return render_template('home.html')   # Render the home.html template
    else:
        data=CustomData(   # Create an instance of CustomData to process the input data
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()  # Convert the input data into a DataFrame for prediction
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline() # Create an instance of PredictPipeline to handle the prediction process
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)  # Run the Flask application on all available IP addresses in debug mode for development purposes
        


