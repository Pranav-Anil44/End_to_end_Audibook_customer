End-to-End Audiobook Customer Prediction
Project Overview

This project predicts whether a customer will purchase again after listening to an audiobook. It’s a complete end-to-end machine learning pipeline starting from raw data, through preprocessing and exploratory analysis, to multiple model implementations (Logistic Regression and Deep Learning).

The project demonstrates:

Handling real-world customer behavioral data
Performing EDA and outlier treatment
Applying feature scaling and preprocessing
Training and evaluating classification models
Comparing Logistic Regression vs Deep Learning performance

📂 Repository Structure
End_to_end_Audiobook_customer/
│
├── Datasets/                         
│   ├── Audiobook_initial.csv
│   ├── Audiobooks_data.csv
│   ├── audiobook_outlier_treated.csv
│
├── Jupyter_notebooks/                
│   ├── 01_Initial_check_Audiobook.ipynb
│   ├── 02_Outlier_treatment_Audiobook.ipynb
│   ├── 03_Feature_scaling_Audiobook.ipynb
│   ├── 04_Model_Logistic_Regression_Audiobook.ipynb
│   ├── 05_Model_DeepLearning_Audiobook.ipynb
│

└── README.md

Tech Stack

Python

Libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn, tensorflow/keras

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




Pranav A Kumar

https://www.linkedin.com/in/pranav-a-kumar-2a39b4358/
https://github.com/Pranav-Anil44


