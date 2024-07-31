from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

# Initialize Flask app
app = Flask(__name__)

# Define prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data, index=[0])
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
