# Loan-Eligibility-Prediction
-End to End Implementation and deployment of Loan eligibility prediction web app

-This project aims to predict loan approval using machine learning techniques. It involves building a predictive model based on a dataset containing various features such as applicant income, credit history, property area, etc., to determine whether a loan application should be approved or not.

## Overview

The project follows these main steps:

1. **Data Collection**: Obtain the dataset containing information about loan applicants, including various features such as income, credit history, and property area.

2. **Exploratory Data Analysis (EDA)**: Perform data exploration to understand the dataset's structure, identify patterns, and gain insights into the relationships between different features.

3. **Data Preprocessing**: Cleanse and preprocess the dataset by handling missing values, encoding categorical variables, and scaling numerical features.

4. **Model Training**: Train a machine learning model (Random Forest Classifier in this case) on the preprocessed dataset to predict loan approval.

5. **Model Evaluation**: Evaluate the performance of the trained model using various metrics such as accuracy, precision, recall, and F1-score.

6. **Deployment as Web App**: Deploy the trained model as a web application using Streamlit, allowing users to input their data and receive loan approval predictions.

## Dataset

The dataset used in this project contains information about loan applicants, including the following features:

- `Gender`: Gender of the applicant (Male/Female)
- `Married`: Marital status of the applicant (Yes/No)
- `Dependents`: Number of dependents
- `Education`: Education level of the applicant (Graduate/Not Graduate)
- `Self_Employed`: Whether the applicant is self-employed (Yes/No)
- `ApplicantIncome`: Income of the applicant
- `CoapplicantIncome`: Income of the co-applicant
- `LoanAmount`: Loan amount in thousands
- `Loan_Amount_Term`: Term of the loan in months
- `Credit_History`: Credit history meets guidelines (1 for Yes, 0 for No)
- `Property_Area`: Area of property (Rural, Semiurban, Urban)
- `Loan_Status`: Loan approval status (1 for Approved, 0 for Not Approved)

## Deployment

The trained model is deployed as a web application using Streamlit. Users can input their data through the web interface, and the application will predict whether their loan application would be approved or not based on the model's predictions.

To run the web app locally, follow these steps:

1. Install the required libraries specified in the `requirements.txt` file.
2. Run the Streamlit app using the command `streamlit run app.py`.
3. Access the web app in your browser using the provided URL.

## Dependencies

The project uses the following Python libraries:

- `pandas`
- `numpy`
- `scikit-learn`
- `streamlit`

These dependencies can be installed using `pip` by running `pip install -r requirements.txt`.