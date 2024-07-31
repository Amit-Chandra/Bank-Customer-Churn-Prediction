from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load the model and transformers
model = joblib.load('churn_model.pkl')

@app.route('/')
def index():
    return "Welcome to the Churn Prediction API!"


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the JSON data from the request
        data = request.get_json()
        
        # Create a DataFrame from the JSON data
        input_data = pd.DataFrame(data)
        
        # Ensure the input features match those used for training
        input_data = input_data[model.named_steps['preprocessor'].transformers_[0][2] + model.named_steps['preprocessor'].transformers_[1][2]]
        
        # Make prediction
        prediction = model.predict(input_data)
        
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)






















# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# app = Flask(__name__)

# # Load the model, scaler, and encoder
# model = joblib.load('churn_model.pkl')
# scaler = joblib.load('scaler.pkl')
# encoder = joblib.load('encoder.pkl')

# @app.route('/')
# def index():
#     return "Welcome to the Churn Prediction API!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Load the JSON data from the request
#         data = request.get_json()
        
#         # Create a DataFrame from the JSON data
#         input_data = pd.DataFrame(data)
        
#         # Separate the categorical and numerical features
#         categorical_features = ['country', 'gender']
#         numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 
#                               'credit_card', 'active_member', 'estimated_salary']
        
#         # Debug: Print input data
#         print("Original Input Data:")
#         print(input_data)
        
#         # Encode categorical features
#         if len(input_data[categorical_features].dropna()) > 0:
#             encoded_features = encoder.transform(input_data[categorical_features])
#             encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
#         else:
#             encoded_df = pd.DataFrame(columns=encoder.get_feature_names_out(categorical_features))
        
#         # Drop the original categorical features
#         input_data = input_data.drop(columns=categorical_features, errors='ignore')
        
#         # Concatenate encoded features with the remaining features
#         input_data = pd.concat([input_data, encoded_df], axis=1)
        
#         # Debug: Print data before scaling
#         print("Data Before Scaling:")
#         print(input_data)
        
#         # Ensure the columns are in the same order as when the scaler was fitted
#         expected_columns = numerical_features + list(encoder.get_feature_names_out(categorical_features))
#         input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        
#         # Debug: Print data after reindexing
#         print("Data After Reindexing:")
#         print(input_data)
        
#         # Apply the scaler
#         scaled_input_data = scaler.transform(input_data)
        
#         # Debug: Print data after scaling
#         print("Data After Scaling:")
#         print(pd.DataFrame(scaled_input_data, columns=expected_columns))
        
#         # Make prediction
#         prediction = model.predict(scaled_input_data)
        
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         # Debug: Print the error message
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

































# from flask import Flask, request, jsonify
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.ensemble import RandomForestClassifier
# import joblib

# app = Flask(__name__)

# # Load the model, scaler, and encoder
# model = joblib.load('churn_model.pkl')
# scaler = joblib.load('scaler.pkl')
# encoder = joblib.load('encoder.pkl')

# @app.route('/')
# def index():
#     return "Welcome to the Churn Prediction API!"

# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         # Load the JSON data from the request
#         data = request.get_json()
        
#         # Create a DataFrame from the JSON data
#         input_data = pd.DataFrame(data)
        
#         # Separate the categorical and numerical features
#         categorical_features = ['country', 'gender']
#         numerical_features = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 
#                               'credit_card', 'active_member', 'estimated_salary']
        
#         # Debug: Print input data
#         print("Original Input Data:")
#         print(input_data)
        
#         # Encode categorical features
#         if not input_data[categorical_features].dropna().empty:
#             encoded_features = encoder.transform(input_data[categorical_features])
#             encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
#         else:
#             encoded_df = pd.DataFrame(columns=encoder.get_feature_names_out(categorical_features))
        
#         # Debug: Print encoded features
#         print("Encoded Features:")
#         print(encoded_df)
        
#         # Drop the original categorical features
#         input_data = input_data.drop(columns=categorical_features, errors='ignore')
        
#         # Concatenate encoded features with the remaining features
#         input_data = pd.concat([input_data, encoded_df], axis=1)
        
#         # Ensure the columns are in the same order as when the scaler was fitted
#         expected_columns = numerical_features + list(encoder.get_feature_names_out(categorical_features))
#         input_data = input_data.reindex(columns=expected_columns, fill_value=0)
        
#         # Debug: Print final input data before scaling
#         print("Data Before Scaling:")
#         print(input_data)
        
#         # Apply the scaler
#         input_data = scaler.transform(input_data)
        
#         # Make prediction
#         prediction = model.predict(input_data)
        
#         return jsonify({'prediction': prediction.tolist()})
#     except Exception as e:
#         # Debug: Print the error message
#         print(f"Error: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
