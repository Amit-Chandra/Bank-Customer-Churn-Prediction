# Customer Churn Prediction

## Overview

This project aims to predict customer churn for a bank using machine learning techniques. Churn prediction is essential for retaining customers and enhancing the overall business strategy. This project involves data preprocessing, model building, evaluation, and deployment of a churn prediction system.

## Project Features

1. **Data Processing and Cleaning**
   - Handle missing values and outliers.
   - Encode categorical features and scale numerical features.
   
2. **Exploratory Data Analysis (EDA)**
   - Visualize the distribution of various features using histograms and count plots.
   - Analyze correlations between features using heatmaps.
   
3. **Model Building**
   - Implement several machine learning models, including Random Forest and Logistic Regression.
   - Apply ensemble techniques like Voting Classifier for improved performance.
   
4. **Model Evaluation**
   - Assess model performance using metrics like accuracy, precision, recall, and F1-score.
   - Use cross-validation to ensure model robustness.

5. **Hyperparameter Tuning**
   - Optimize model performance using Grid Search and Random Search techniques.

6. **Feature Engineering**
   - Create new features based on existing ones to capture more complex relationships.

7. **Deployment**
   - Develop a Flask application for real-time churn prediction.
   - Provide an API endpoint for receiving customer data and returning churn predictions.

8. **Interactive Visualization**
   - Generate detailed and interactive visualizations to understand the data and model predictions.

## Installation

To set up this project, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Amit-Chandra/Bank-Customer-Churn-Prediction.git
   cd Bank-Customer-Churn-Prediction
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**

   ```bash
   pip install -r requirements.txt
   ```

## Data

The dataset used for this project is `bank_customer_churn_prediction.csv`, which includes the following features:

- `customer_id`: Unique identifier for each customer.
- `credit_score`: Credit score of the customer.
- `country`: Country where the customer resides.
- `gender`: Gender of the customer.
- `age`: Age of the customer.
- `tenure`: Number of years the customer has been with the bank.
- `balance`: Account balance of the customer.
- `products_number`: Number of products the customer has with the bank.
- `credit_card`: Whether the customer has a credit card (1 = Yes, 0 = No).
- `active_member`: Whether the customer is an active member (1 = Yes, 0 = No).
- `estimated_salary`: Estimated salary of the customer.
- `churn`: Target variable indicating whether the customer churned (1 = Yes, 0 = No).

## Usage

### Training and Evaluation

To train and evaluate the models, run the following script:

```bash
python train_and_evaluate.py
```

This script performs the following tasks:
- Loads and preprocesses the data.
- Trains multiple models.
- Evaluates model performance and selects the best model.

### Deployment

To run the Flask application for churn prediction, use the following command:

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`. You can send a POST request to the `/predict` endpoint with customer data in JSON format:

```json
{
    "credit_score": [650],
    "country": ["France"],
    "gender": ["Female"],
    "age": [30],
    "tenure": [5],
    "balance": [50000],
    "products_number": [2],
    "credit_card": [1],
    "active_member": [1],
    "estimated_salary": [60000]
}
```

### Interactive Visualization

To generate interactive visualizations, run:

```bash
python customer_churn_prediction_and_analysis.ipynb
```

This script creates various plots to visualize data distributions, correlations, and model predictions.

## Contribution

Contributions to this project are welcome. To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

- Data Source: Kaggle
- Libraries: Scikit-learn, Pandas, Seaborn, Matplotlib, Flask, etc.

## Here are some pictures of Visualization


![Visualization of Correlation Heatmap](https://github.com/Amit-Chandra/Bank-Customer-Churn-Prediction/blob/main/1.png)
![3D Visualization of Credit Card Holders](https://github.com/Amit-Chandra/Bank-Customer-Churn-Prediction/blob/main/2.png)
![Visualization of Credit Card Holders](https://github.com/Amit-Chandra/Bank-Customer-Churn-Prediction/blob/main/3.png)