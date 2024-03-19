import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders for categorical features
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Load the scalers for numeric features
with open('scalers.pkl', 'rb') as file:
    scalers = pickle.load(file)


# Streamlit app
def main():
    st.title('Loan Approval Predictor')

    # Input form
    st.sidebar.header('Input Features')

    # Collect user input
    user_input = {}
    user_input['Gender'] = st.sidebar.radio('Gender', ['Male', 'Female'])
    user_input['Married'] = st.sidebar.radio('Married', ['Yes', 'No'])
    user_input['Dependents'] = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
    user_input['Education'] = st.sidebar.radio('Education', ['Graduate', 'Not Graduate'])
    user_input['Self_Employed'] = st.sidebar.radio('Self Employed', ['Yes', 'No'])
    user_input['ApplicantIncome'] = st.sidebar.number_input('Applicant Income', value=0, step=1)
    user_input['CoapplicantIncome'] = st.sidebar.number_input('Coapplicant Income', value=0, step=1)
    user_input['LoanAmount'] = st.sidebar.number_input('Loan Amount', value=0, step=1)
    user_input['Loan_Amount_Term'] = st.sidebar.number_input('Loan Amount Term', value=0, step=1)
    user_input['Credit_History'] = st.sidebar.selectbox('Credit History', [0, 1])
    user_input['Property_Area'] = st.sidebar.radio('Property Area', ['Rural', 'Semiurban', 'Urban'])

    # Submit button
    if st.sidebar.button('Submit'):
        # Preprocess user input
        input_df = pd.DataFrame([user_input])

        # Transform the user input data using the label encoders fitted during training
        for feature in label_encoders:
            input_df[feature] = label_encoders[feature].transform(input_df[feature])

        # Scale numeric features using scalers fitted during training
        for feature in scalers:
            input_df[feature] = scalers[feature].transform(input_df[feature].values.reshape(-1, 1))

        # Make prediction
        prediction = model.predict(input_df)

        # Display prediction
        st.subheader('Prediction')
        if prediction[0] == 1:
            st.write('Congratulations! Your loan application is approved.')
        else:
            st.write('Sorry, your loan application is rejected.')


if __name__ == '__main__':
    main()