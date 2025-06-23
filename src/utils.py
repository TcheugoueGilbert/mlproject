import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):  # Save the object to a file using dill
    try:
        dir_path = os.path.dirname(file_path)  # Get the directory path from the file path

        os.makedirs(dir_path, exist_ok=True)   # Create the directory if it does not exist

        with open(file_path, "wb") as file_obj:  # Open the file in write-binary mode
            pickle.dump(obj, file_obj)  # Save the object to the file

    except Exception as e:
        raise CustomException(e, sys)  # Custom exception to handle errors during object saving

def evaluate_models(X_train, y_train,X_test,y_test,models,param): # Evaluate multiple models using GridSearchCV for hyperparameter tuning
    try:
        report = {}

        for i in range(len(list(models))):  # Iterate through each model
            model = list(models.values())[i]  # Get the model from the dictionary
            para=param[list(models.keys())[i]]  

            gs = GridSearchCV(model,para,cv=3)  # Initialize GridSearchCV with the model and parameters
            gs.fit(X_train,y_train) # Fit the model to the training data

            model.set_params(**gs.best_params_)  # Set the best parameters found by GridSearchCV to the model
            model.fit(X_train,y_train)  # Train the model on the training data

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score   # Store the test score in the report dictionary

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
