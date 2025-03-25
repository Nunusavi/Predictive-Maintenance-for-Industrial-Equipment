from flask import Flask, render_template, request, jsonify, send_file
import os
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
import uuid

model = joblib.load('../Predictive_maintenance_model V3.pkl')


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
    
    data.drop(columns=['Product ID'], inplace=True)
    # add Temprature_Diffrence column
    data['Temprature_Diffrence'] = data['Process temperature [K]'] - data['Air temperature [K]']

    type_mapping = {'L': 1, 'M': 2, 'H': 3}
    data['Type'] = data['Type'].map(type_mapping)
    
    return data


app = Flask(__name__, static_url_path='/static', static_folder='static')

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = handle_file_upload()
    if isinstance(data, str):
        return jsonify({'error': data}), 400
    
    try:
        # Preprocess the data
        data = preprocess_data(data)
        
        # Make predictions
        prediction = model.predict(data)
        
        # Convert predictions into a DataFrame and merge with input data
        df = pd.DataFrame(prediction, columns=['Prediction'])
        df = pd.merge(data, df, left_index=True, right_index=True)
        
        # Generate a unique filename
        output_file = f'prediction_{uuid.uuid4().hex}.xlsx'
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, index=False)
        
        # Save the filename in the session or return it in the response
        return jsonify({'success': True, 'message': 'File processed successfully!', 'filename': output_file})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download', methods=['GET'])
def download():
    # Get the filename from the query parameter
    output_file = request.args.get('filename')
    if output_file and os.path.exists(output_file):
        return send_file(
            output_file,
            as_attachment=True,
            download_name='prediction.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    app.run(debug=True)