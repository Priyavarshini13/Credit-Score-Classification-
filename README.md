# Credit-Score-Classification-
AIML project for credit score classification
Credit Score Classification
Project Overview

This project focuses on predicting whether a customer has a good or bad credit score using machine learning models. By analyzing financial and personal information, the models help identify high-risk customers for banks and financial institutions. The pipeline includes data preprocessing, feature engineering, handling class imbalance, and model evaluation. Multiple models were tested, including Logistic Regression, SVM, Random Forest, and XGBoost, with Logistic Regression + SMOTE achieving the best balance of accuracy and recall. Visualizations such as feature importance, confusion matrices, and ROC curves provide insights into model performance and decision-making.

Tech Stack

Language: Python

Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn, XGBoost

Environment: Google Colab / Jupyter Notebook

Dataset

Contains 1000 customer records with 20+ features.

Target variable: Credit Score (1 = Good, 2 = Bad)

No missing values in the dataset

Project Steps
1️⃣ Exploratory Data Analysis (EDA)

Checked class distribution: 70% good credit, 30% bad credit

Visualized feature distributions and correlations

Identified imbalanced target class

2️⃣ Preprocessing & Feature Engineering

Encoded categorical features using One-Hot Encoding

Scaled numeric features using StandardScaler

Train-test split (80% train, 20% test)

Target mapped to 0 = Good Credit, 1 = Bad Credit for ML models

3️⃣ Model Training & Evaluation

Tested multiple models:

Model	Accuracy	Recall (Bad Credit)	Notes
Logistic Regression (Basic)	0.78	0.53	Biased toward majority class
Logistic Regression (Class Weights)	0.75	0.80	Improved minority recall
Logistic Regression + SMOTE	0.74	0.78	Balanced fairness (Best model)
SVM (Class Weights)	0.76	0.77	Improved minority recall
Random Forest + SMOTE	0.77	0.50	High precision but poor minority recall
XGBoost	0.76	0.52	Good overall but minority recall low

Best Performing Model: Logistic Regression + SMOTE

Achieved balanced accuracy and high recall for bad credit, making it reliable for identifying high-risk customers.

4️⃣ Visualizations

Class Distribution Pie Chart – Shows proportion of good vs bad credit

Top 15 Feature Importances (Random Forest) – Identifies most predictive features

Confusion Matrices – For all six models, highlighting performance

ROC-AUC Curves – Visual comparison of model performance

How to Run

Clone the repo:

git clone https://github.com/your-username/Credit-Score-Classification-.git
cd Credit-Score-Classification-


Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook Credit_Score_Classification.ipynb


Run all cells to reproduce results.

Conclusion

Logistic Regression + SMOTE is the most reliable model for predicting bad credit.

Incorporating SMOTE helped to balance class imbalance and improve minority recall.

Visualizations provide insight into feature importance and model performance, making it a complete and interpretable pipeline.
