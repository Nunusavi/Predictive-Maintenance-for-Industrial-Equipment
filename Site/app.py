import gradio as gr
import pandas as pd
import joblib

# Load the predictive maintenance model
model = joblib.load('../Predictive_maintenance_model V3.pkl')

def preprocess_data(data):
    # handle missing values
    data.dropna(inplace=True)
    
    data.drop(columns=['Product ID'], inplace=True)

    type_mapping = {'L': 1, 'M': 2, 'H': 3}
    data['Type'] = data['Type'].map(type_mapping)
    
    return data

def predict_maintenance(data):
    # Make predictions using the loaded model
    predictions = model.predict(data)
    return predictions

def predict_with_preprocessing(input_file):
    # Load the input CSV or Excel file
    if input_file.name.endswith('.csv'):
        data = pd.read_csv(input_file)
    elif input_file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(input_file)
    else:
        return "Unsupported file format. Please upload a CSV or Excel file."

    # Preprocess the data
    preprocessed_data = preprocess_data(data)
    
    # Make predictions
    predictions = predict_maintenance(preprocessed_data)
    
    return predictions

# Create the Gradio interface
input_interface = gr.File(label="Upload CSV or Excel file")
output_interface = gr.Textbox(label="Predictions")

gr.Interface(fn=predict_with_preprocessing, inputs=input_interface, outputs=output_interface, title="Predictive Maintenance Demo").launch()
