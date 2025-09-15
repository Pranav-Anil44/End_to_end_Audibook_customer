End-to-End Audiobook Customer Prediction
Project Overview

This project predicts whether a customer will purchase again after listening to an audiobook. It’s a complete end-to-end machine learning pipeline starting from raw data, through preprocessing and exploratory analysis, to multiple model implementations (Logistic Regression and Deep Learning).

The project demonstrates:

Handling real-world customer behavioral data
Performing EDA and outlier treatment
Applying feature scaling and preprocessing
Training and evaluating classification models
Comparing Logistic Regression vs Deep Learning performance

EndtoEnd_ML_Audiobook_project/
│
├── Jupyter_notebooks/         # Training notebooks
│   ├── Model_logistic_Regression_audiobook.ipynb
│   ├── Model_deeplearning_Audiobook.ipynb
│   ├── Outlier_treatment_Audiobook.ipynb
│   └── ...
│
├── Datasets/                  # Input data
│   ├── audiobook_outlier_treated.csv
│   └── audioboook_scaled.csv
│
├── Models/                    # Saved models
│   ├── logistic_model.pkl
│   └── deep_learning_model.h5
│
├── app_logistic.py            # Flask API for Logistic Regression
├── app_deeplearning.py        # Flask API for Deep Learning
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation


Tech Stack

Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, tensorflow/keras,Flask

 Workflow

Initial Data Check & EDA
Understand dataset structure
Identify missing values & distributions
Outlier Treatment
Remove/adjust extreme values using statistical methods
Feature Scaling
Apply normalization/standardization for model stability
Modeling
Logistic Regression (baseline model)
Deep Learning (dense neural network classifier)
Evaluation
Accuracy, Precision, Recall, F1-score
ROC Curve, AUC comparison

Results

Logistic Regression achieved 93% accuracy but F1 score is only 47%

Deep Learning model accuracy was 88% but we can see a significant improvement in F1 score to 63%
We have used SMOTE to reduce the inbalance in the daseset and thus we acheived this result.

Insights: Customer engagement features like Minutes Listened and Completion Percentage were strong predictors.

Models
Logistic Regression
 Trained on scaled features
 Serves as baseline model

Deep Learning
 hidden layers with ReLUDropout for regularization
 Sigmoid output for binary classification

Using the Deployed Models

Both models are deployed as Flask APIs.

1️) Run the APIs

Open two terminals:

# Terminal 1: Logistic Regression API
python app_logistic.py

# Terminal 2: Deep Learning API
python app_deeplearning.py

You’ll see:
Running on http://127.0.0.1:5000

2️) Test Logistic Regression API

Endpoint:
POST http://127.0.0.1:5000/predict

Request example:

import requests
url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

data = {
    "Book_length_min": 300,
    "Book_length_char": 150000,
    "Avg_rating": 4.5,
    "Rating_count": 1200,
    "Review_score": 8,
    "Price": 20,
    "Discount": 2,
    "Total_minutes_listened": 150,
    "Completion": 60,
    "Support_requests": 0,
    "Support_request": 0
}

response = requests.post(url, headers=headers, json=data)
print(response.json())


Example response:
{"LogisticRegression_Prediction": 1}

Test Deep Learning API

Endpoint:
POST http://127.0.0.1:5000/predict


Request example:

import requests
url = "http://127.0.0.1:5000/predict"
headers = {"Content-Type": "application/json"}

data = {
    "Book_length_min": 300,
    "Book_length_char": 150000,
    "Avg_rating": 4.5,
    "Rating_count": 1200,
    "Review_score": 8,
    "Price": 20,
    "Discount": 2,
    "Total_minutes_listened": 150,
    "Completion": 60,
    "Support_requests": 0,
    "Support_request": 0
}

response = requests.post(url, headers=headers, json=data)
print(response.json())

Example response:

{
  "DeepLearning_Prediction": 1,
  "Probability": 0.8734
}

 Notes

JSON keys must match the training dataset feature names.
Logistic Regression returns only a binary prediction (0/1).
Deep Learning returns prediction and probability.

Pranav A Kumar
https://www.linkedin.com/in/pranav-a-kumar-2a39b4358/
https://github.com/Pranav-Anil44


