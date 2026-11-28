# employee-attrition-prediction
# Employee Attrition Prediction Using Machine Learning

## 1. Project Overview

Employee Attrition Prediction is a Machine Learning project designed to predict whether an employee is likely to leave an organization based on Human Resources (HR) data.

The project uses the IBM HR Analytics Employee Attrition dataset and applies supervised learning, unsupervised learning, ensemble learning, and hyperparameter optimization techniques.

A web application was also developed to allow users to enter employee information and receive real-time attrition predictions.

## 2. Objectives

- Analyze employee HR data to identify attrition-related patterns.
- Perform data preprocessing and feature engineering.
- Build and evaluate supervised machine learning models.
- Apply unsupervised learning techniques for employee clustering.
- Implement an ensemble learning algorithm.
- Optimize model performance using cross-validation and GridSearchCV.
- Deploy a machine learning prediction application online.

## 3. Dataset Description

**Dataset:** IBM HR Analytics Employee Attrition Dataset

- Total records: 1,470 employees
- Total original features: 35
- Target variable: `Attrition`
- Target classes: `Yes` and `No`

### Important Features

| Feature | Description |
|---|---|
| Age | Employee age |
| MonthlyIncome | Monthly salary |
| JobSatisfaction | Job satisfaction level |
| OverTime | Whether the employee works overtime |
| Department | Employee department |
| JobRole | Employee job role |
| YearsAtCompany | Years spent at the company |
| WorkLifeBalance | Employee work-life balance |

## 4. Machine Learning Workflow

### Step 1: Data Loading and Exploration

- Imported the dataset using Pandas.
- Examined the first and last records using `head()` and `tail()`.
- Checked dataset dimensions using `shape`.
- Inspected column information using `info()`.
- Identified missing values using `isnull().sum()`.
- Examined the distribution of the target variable.

### Step 2: Data Preprocessing

The following preprocessing operations were performed:

- Removed unnecessary and constant columns.
- Separated independent features (X) and target variable (y).
- Converted the target variable into numerical values:
  - Yes = 1
  - No = 0
- Encoded categorical variables using OneHotEncoder.
- Scaled numerical features using StandardScaler.
- Split the dataset into training and testing sets.

**Train-test split:**
- Training data: 80% (1,176 records)
- Testing data: 20% (294 records)

### Step 3: Supervised Learning

Two supervised machine learning algorithms were implemented.

**A. Logistic Regression**

Logistic Regression was used as a baseline classification model to predict employee attrition.

Evaluation metrics:
- Accuracy: 75.17%
- Recall: 63.83%
- F1 Score: 45.11%

**B. Decision Tree Classifier**

A Decision Tree was implemented as the second supervised learning model.

Evaluation metrics:
- Accuracy: 77.89%
- Recall: 29.79%
- F1 Score: 30.11%

### Step 4: Unsupervised Learning

Unsupervised learning techniques were applied to explore employee groupings and patterns.

**A. K-Means Clustering**

- Applied K-Means to group employees into clusters.
- Examined employee characteristics within each cluster.
- Evaluated clustering performance using the Silhouette Score.

Silhouette Score: 0.1109

**B. DBSCAN Clustering**

- Applied DBSCAN for density-based clustering.
- Examined the resulting cluster distribution.

Result: All 1,470 records were classified as noise (-1) with the initial DBSCAN configuration. Therefore, this run did not produce meaningful clusters.

### Step 5: Ensemble Learning – Random Forest

Random Forest Classifier was implemented as an ensemble learning technique using bagging.

It combines multiple decision trees to improve classification performance and reduce dependence on a single tree.

### Step 6: Hyperparameter Optimization

GridSearchCV was used to identify suitable Random Forest hyperparameters.

**Optimization configuration:**
- Cross-validation: 5-fold
- Scoring metric: F1 Score
- Search method: GridSearchCV

**Best parameters obtained:**

- `max_depth`: 5
- `min_samples_split`: 2
- `n_estimators`: 100

Best cross-validation F1 Score: 0.50785

### Step 7: Model Evaluation

The optimized Random Forest model was evaluated using the held-out test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 81% |
| Recall (Attrition = Yes) | 57% |
| F1 Score (Attrition = Yes) | 49% |
| Precision (Attrition = Yes) | 42% |

The optimized Random Forest achieved a better test F1 Score than the baseline Logistic Regression model.

## 5. Model Saving and Deployment

The trained models were saved using Joblib.

Two model artifacts were created:

- `employee_attrition_model.pkl` – optimized full-feature Random Forest model.
- `employee_attrition_app.pkl` – compact four-feature Logistic Regression model used by the web application.

The compact deployment model uses the following inputs:

1. Age
2. Monthly Income
3. Job Satisfaction
4. OverTime

**Note:** The deployed four-input model is a separate demonstration model, not the optimized full-feature Random Forest model.

## 6. Web Application Architecture

The application was converted from Streamlit to a Flask-based web application for Vercel deployment.

### Architecture

User Input → Web Interface → Flask Prediction API → Trained ML Model → Prediction Result

### Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Machine learning algorithms |
| Matplotlib / Seaborn | Data visualization |
| Joblib | Model serialization |
| Flask | Backend and prediction API |
| HTML | Web interface |
| CSS | Application styling |
| JavaScript | API communication |
| Git and GitHub | Version control |
| Vercel | Application hosting |

## 7. Project Structure

```text
employee-attrition-prediction/
│
├── app.py
├── streamlit_app.py
├── requirements.txt
├── .python-version
├── employee_attrition_app.pkl
├── employee_attrition_model.pkl
├── pythonkaggledataset.py
└── README.md
```

## 8. Local Execution

### Clone the repository

```bash
git clone https://github.com/VicTheDev330/employee-attrition-prediction.git
cd employee-attrition-prediction
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the Flask application

```bash
python -m flask --app app run
```

Open the application at:

```text
http://127.0.0.1:5000
```

## 9. Live Deployment

The application is deployed using Vercel.

**Live Application:** Add your Vercel deployment URL here.

**GitHub Repository:**  
https://github.com/VicTheDev330/employee-attrition-prediction

## 10. Conclusion

This project demonstrates the application of machine learning techniques to an HR analytics problem.
<img width="1920" height="951" alt="Screenshot (703)" src="https://github.com/user-attachments/assets/fa5c72dc-c376-4181-ae67-c0fad5dce36d" />
<img width="1920" height="1080" alt="Screenshot (706)" src="https://github.com/user-attachments/assets/dcf9c493-b3b8-40c7-99b4-40cc3b693d44" />

It includes data preprocessing, supervised classification, unsupervised clustering, ensemble learning, hyperparameter optimization, model evaluation, and web application deployment.

The project also demonstrates how a trained machine learning model can be integrated with a Flask backend and made accessible through an online application.

**Disclaimer:** Predictions are statistical estimates based on the training dataset and should not be treated as definitive judgments about individual employees.
