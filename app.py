from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)


model = joblib.load('churn_model.pkl')

@app.route('/')
def index():
    return "Welcome to the Churn Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json()
               
        input_data = pd.DataFrame(data)
        
        input_data = input_data[model.named_steps['preprocessor'].transformers_[0][2] + model.named_steps['preprocessor'].transformers_[1][2]]
        
        prediction = model.predict(input_data)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

















