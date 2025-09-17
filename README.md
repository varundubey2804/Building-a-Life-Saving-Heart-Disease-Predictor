# Building-a-Life-Saving-Heart-Disease-Predictor
Heart Disease Prediction using Machine Learning
Project Overview

This project aims to predict the presence and severity of heart disease using the UCI Heart Disease dataset. By leveraging machine learning techniques, the goal is to analyze key health indicators, identify important features, and build predictive models that can assist in early diagnosis and decision support for healthcare professionals.

Dataset

Source: Kaggle - Heart Disease UCI Dataset

File Used: heart_disease_uci.csv

Target Variable: num

0 → No Disease

1–4 → Different levels of heart disease severity

Exploratory Data Analysis (EDA)

Distribution of heart disease across the dataset

Visualizations of age, chest pain type, sex, and maximum heart rate against the target variable

Correlation heatmap for numerical features

Detection of missing values and handling strategies

Data Preprocessing

Numerical Features: Imputation (mean), Standard Scaling

Categorical Features: Imputation (most frequent), One-Hot Encoding

ColumnTransformer & Pipelines used for seamless preprocessing

Machine Learning Models

The following models were trained and evaluated:

Logistic Regression

Random Forest Classifier

Support Vector Machine (SVM)

K-Nearest Neighbors (KNN)

Evaluation Metrics

Accuracy

Precision, Recall, F1-score

Confusion Matrix

Results and Insights

Random Forest revealed the most influential features in prediction.

SVM showed strong classification performance across multiple classes.

Key health indicators such as chest pain type, age, sex, cholesterol, and max heart rate were significant in predicting heart disease.

Visualizations

Distribution of heart disease cases

Feature-target relationships (age, chest pain, sex, max heart rate)

Correlation heatmap of numerical variables

Confusion matrix for SVM

Top 10 most important features from Random Forest

How to Run the Project

Clone the repository:

git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction


Install dependencies:

pip install -r requirements.txt


Run the notebook/script to train and evaluate models:

python heart_disease_prediction.py

Project Structure
├── data/  
│   └── heart_disease_uci.csv  
├── heart_disease_prediction.py  
├── README.md  
└── requirements.txt  

Future Work

Hyperparameter tuning for improved model performance

Implementation of deep learning models

Deployment of the best-performing model as a web app or API

Technologies Used

Python

Pandas, NumPy, Matplotlib, Seaborn

Scikit-learn

Acknowledgments

Dataset provided by UCI Machine Learning Repository
 via Kaggle
