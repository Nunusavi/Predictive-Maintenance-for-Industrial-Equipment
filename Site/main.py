from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import joblib
from pandas import json_normalize

from werkzeug.utils import secure_filename

model = joblib.load('../Predictive_maintenance_model V2.pkl')


def handle_file_upload():
    file = request.files['file']
    
    if not file:
        return 'No file provided', 400
    if file.filename.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.filename.endswith('.xlsx'):
        data = pd.read_excel(file)
    elif file.filename.endswith('.xls'):
        data = pd.read_excel(file)
    elif file.filename.endswith('.json'):
        data = pd.read_json(file)
    else:
        return 'Invalid file type, Please upload a CSV file, Excel or JSON file', 400
    return data

def preprocess_data(data):
    # handle missing values
    data.dropna(inplace=True)
    
    data.drop(columns=['Machine failure'], inplace=True)
    data.drop(columns=['Product ID'], inplace=True)

    type_mapping = {'L': 1, 'M': 2, 'H': 3}
    data['Type'] = data['Type'].map(type_mapping)
    
    return data

# take prediction data and create a table that is displayed on the website


app = Flask(__name__,static_url_path='/static', static_folder='static')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = handle_file_upload()
    if isinstance(data, str):
        return data
    data = preprocess_data(data)
    prediction = model.predict(data)
    result = jsonify({'prediction': prediction.tolist()})
    table = create_table(prediction)
    # convert the prediction into a dataframe and merge it with the data 
    df = pd.DataFrame(prediction, columns=['Prediction'])
    df = pd.merge(data, df, left_index=True, right_index=True)
    # convert df to excel
    df.to_excel('prediction.xlsx', index=False)
    # download the excel file
    return send_file('prediction.xlsx', as_attachment=True)


# function that takes result of prediction and creates a table that is displayed on the website
def create_table(prediction):
    table = '<table class="table"><thead><tr><th scope="col">Machine ID</th><th scope="col">Prediction</th></tr></thead><tbody>'
    for i, pred in enumerate(prediction):
        table += f'<tr><td>{i}</td><td>{pred}</td></tr>'
    table += '</tbody></table>'
    return table


    
if __name__ == '__main__':
    app.run(debug=True)