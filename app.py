import os
import pandas as pd
from flask import Flask, render_template, request
import pickle
from werkzeug.utils import redirect
from flask_mysqldb import MySQL
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import sweetviz as sv

app = Flask(__name__, static_url_path='/static')

# Setup for CLASSIFICATION

# Loading in classification model
employee_path = 'employee.pkl'
with open(employee_path, 'rb') as model_file_employee:
    model_employee = pickle.load(model_file_employee)

# Load the Employee dataset
filename_employee = 'employee_data.csv'
dataframe_employee = pd.read_csv(filename_employee)

# Separate features input and output
x_employee = dataframe_employee.drop('LeaveOrNot', axis=1)
y_employee = dataframe_employee['LeaveOrNot']

# Convert categorical variables into numerical format using one-hot encoding
x_encoded_employee = pd.get_dummies(x_employee)

# Function to make predictions using the Employee Model


def predict_employee_class(features):
    # Convert the input features into a DataFrame
    input_data = pd.DataFrame([features], columns=x_employee.columns)

    # One-hot encode the input data with the columns used during training
    input_data_encoded = pd.get_dummies(input_data).reindex(
        columns=x_encoded_employee.columns, fill_value=0)

    # Use the trained model to make predictions
    prediction = model_employee.predict(input_data_encoded)

    return prediction[0]


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_employee():
    # Get input features from the form
    features = [request.form[field] for field in x_employee.columns]

    # Use the model to make predictions
    predicted_class = predict_employee_class(features)

    # Map predicted class to human-readable labels
    class_labels = {1: 'Leave', 0: 'Not'}
    predicted_label = class_labels.get(predicted_class, 'unknown')

    return render_template('index.html', prediction=predicted_label)


# Setup for REGRESSION

# Loading in regression model
bitcoin_path = 'bitcoin.pkl'
with open(bitcoin_path, 'rb') as model_file_bitcoin:
    model_bitcoin = pickle.load(model_file_bitcoin)

# Load the Bitcoin dataset
filename_bitcoin = 'bitcoin_price.csv'
dataframe_bitcoin = pd.read_csv(filename_bitcoin)

# Separate features input and output
x_bitcoin = dataframe_bitcoin.drop('Close', axis=1)
y_bitcoin = dataframe_bitcoin['Close']

# Function to make predictions using the Bitcoin Model


def predict_bitcoin_reg(features):
    # Convert the input features into a DataFrame
    input_databtc = pd.DataFrame([features], columns=x_bitcoin.columns)

    # Use the trained model to make predictions
    prediction = model_bitcoin.predict(input_databtc)

    return prediction[0]


@app.route('/regression')
def regression():
    return render_template('regression.html')


@app.route('/regression', methods=['POST'])
def predict_bitcoin():
    # Get input features from the form
    featuresbtc = [request.form[field] for field in x_bitcoin.columns]

    # Use the model to make predictions for regression
    predicted_price = predict_bitcoin_reg(featuresbtc)

    # Render the regression.html template with the predicted price
    return render_template('regression.html', predictionbtc=predicted_price)


# Load datasets
# data1 = pd.read_csv('employee_data.csv')
# data2 = pd.read_csv('bitcoin_price.csv')

# Compare datasets
# report = sv.compare([data1, 'Employee Data'], [data2, 'Bitcoin Price Data'])
# report2 = sv.compare([data2, 'Bitcoin Price Data'], [data1, 'Employee Data'])


# @app.route('/eda')
# def eda():
#    return render_template('eda.html', report_html=report.show_html())


# @app.route('/eda2')
# def eda2():
#    return render_template('eda2.html', report2_html=report2.show_html())


if __name__ == '__main__':
    app.run(debug=True)
