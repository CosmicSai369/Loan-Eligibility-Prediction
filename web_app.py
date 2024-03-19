# # # import pickle
# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# # #
# # # # Load the trained model
# # # with open('trained_model.pkl', 'rb') as file:
# # #     model = pickle.load(file)
# # #
# # # # Load the label encoder for categorical features
# # # with open('label_encoder.pkl', 'rb') as file:
# # #     label_encoder = pickle.load(file)
# # #
# # # # Load the standard scaler for numeric features
# # # with open('scaler.pkl', 'rb') as file:
# # #     scaler = pickle.load(file)
# # #
# # #
# # # # Function to preprocess input data
# # # def preprocess_data(input_data):
# # #     # Define categorical and numeric features
# # #     cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
# # #     numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
# # #
# # #     # Convert categorical features using label encoder
# # #     for feature in cat_features:
# # #         # Handle previously unseen labels
# # #         unseen_labels_mask = ~input_data[feature].isin(label_encoder.classes_)
# # #         if unseen_labels_mask.any():
# # #             new_labels = input_data.loc[unseen_labels_mask, feature]
# # #             label_encoder.classes_ = np.concatenate([label_encoder.classes_, new_labels])
# # #         input_data[feature] = label_encoder.transform(input_data[feature])
# # #
# # #     # Scale numeric features using standard scaler
# # #     input_data[numeric_features] = scaler.transform(input_data[numeric_features])
# # #
# # #     return input_data
# # #
# # #
# # # # Streamlit app
# # # def main():
# # #     # Define categorical and numeric features
# # #     cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
# # #     numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
# # #
# # #     st.title('Loan Approval Predictor')
# # #
# # #     # Input form
# # #     st.sidebar.header('Input Features')
# # #
# # #     # Collect user input
# # #     user_input = {}
# # #     user_input['Gender'] = st.sidebar.radio('Gender', ['Male', 'Female'])
# # #     user_input['Married'] = st.sidebar.radio('Married', ['Yes', 'No'])
# # #     user_input['Dependents'] = st.sidebar.selectbox('Dependents', [0, 1, 2, 3])
# # #     user_input['Education'] = st.sidebar.radio('Education', ['Graduate', 'Not Graduate'])
# # #     user_input['Self_Employed'] = st.sidebar.radio('Self Employed', ['Yes', 'No'])
# # #     user_input['ApplicantIncome'] = st.sidebar.number_input('Applicant Income')
# # #     user_input['CoapplicantIncome'] = st.sidebar.number_input('Coapplicant Income')
# # #     user_input['LoanAmount'] = st.sidebar.number_input('Loan Amount')
# # #     user_input['Loan_Amount_Term'] = st.sidebar.number_input('Loan Amount Term')
# # #     user_input['Credit_History'] = st.sidebar.selectbox('Credit History', [0, 1])
# # #     user_input['Property_Area'] = st.sidebar.radio('Property Area', ['Rural', 'Semiurban', 'Urban'])
# # #
# # #     # Preprocess user input
# # #     input_df = pd.DataFrame([user_input])
# # #     input_df = preprocess_data(input_df)
# # #
# # #     # Make prediction
# # #     prediction = model.predict(input_df)
# # #
# # #     # Display prediction
# # #     st.subheader('Prediction')
# # #     if prediction[0] == 1:
# # #         st.write('Congratulations! Your loan application is approved.')
# # #     else:
# # #         st.write('Sorry, your loan application is rejected.')
# # #
# # #
# # # if __name__ == '__main__':
# # #     main()
# #
# # import pickle
# # import streamlit as st
# # import pandas as pd
# # import numpy as np
# # # from sklearn.preprocessing import LabelEncoder, StandardScaler
# #
# # # Load the trained model
# # with open('trained_model.pkl', 'rb') as file:
# #     model = pickle.load(file)
# #
# # # Load the label encoder for categorical features
# # with open('label_encoder.pkl', 'rb') as file:
# #     label_encoder = pickle.load(file)
# #
# # # Load the standard scaler for numeric features
# # with open('scaler.pkl', 'rb') as file:
# #     scaler = pickle.load(file)
# #
# # # Function to preprocess input data
# # def preprocess_data(input_data):
# #     # Define categorical and numeric features
# #     cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
# #     numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
# #
# #     # Convert categorical features using label encoder
# #     for feature in cat_features:
# #         # input_data[feature] = label_encoder.fit_transform(input_data[feature])
# #     # for feature in cat_features:
# #     #     # Handle previously unseen labels
# #     #     unseen_labels_mask = ~input_data[feature].isin(label_encoder.classes_)
# #     #     if unseen_labels_mask.any():
# #     #         new_labels = input_data.loc[unseen_labels_mask, feature]
# #     #         label_encoder.classes_ = np.concatenate([label_encoder.classes_, new_labels])
# #         input_data[feature] = label_encoder.transform(input_data[feature])
# #
# #     # Scale numeric features using standard scaler
# #     input_data[numeric_features] = scaler.transform(input_data[numeric_features])
# #
# #     return input_data
# #
# # # Streamlit app
# # def main():
# #     # Define categorical and numeric features
# #     cat_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Credit_History', 'Property_Area']
# #     numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
# #
# #     st.title('Loan Approval Predictor')
# #
# #     # Input form
# #     st.sidebar.header('Input Features')
# #
# #     # Collect user input
# #     user_input = {}
# #     user_input['Gender'] = st.sidebar.radio('Gender', ['Male', 'Female'])
# #     user_input['Married'] = st.sidebar.radio('Married', ['Yes', 'No'])
# #     user_input['Dependents'] = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
# #     user_input['Education'] = st.sidebar.radio('Education', ['Graduate', 'Not Graduate'])
# #     user_input['Self_Employed'] = st.sidebar.radio('Self Employed', ['Yes', 'No'])
# #     user_input['ApplicantIncome'] = st.sidebar.number_input('Applicant Income', value=0, step=1)
# #     user_input['CoapplicantIncome'] = st.sidebar.number_input('Coapplicant Income', value=0, step=1)
# #     user_input['LoanAmount'] = st.sidebar.number_input('Loan Amount', value=0, step=1)
# #     user_input['Loan_Amount_Term'] = st.sidebar.number_input('Loan Amount Term', value=0, step=1)
# #     user_input['Credit_History'] = st.sidebar.selectbox('Credit History', [0, 1])
# #     user_input['Property_Area'] = st.sidebar.radio('Property Area', ['Rural', 'Semiurban', 'Urban'])
# #
# #     # Submit button
# #     if st.sidebar.button('Submit'):
# #         # Preprocess user input
# #         input_df = pd.DataFrame([user_input])
# #         input_df = preprocess_data(input_df)
# #
# #         # Make prediction
# #         prediction = model.predict(input_df)
# #
# #         # Display prediction
# #         st.subheader('Prediction')
# #         st.write(prediction[0])
# #         # if prediction[0] == 'Y':
# #         #     st.write('Congratulations! Your loan application is approved.')
# #         # else:
# #         #     st.write('Sorry, your loan application is rejected.')
# #
# # if __name__ == '__main__':
# #     main()
# #
#
#
#
#
#
#
#
#
#
#
# #=======================================================================
#
# import pickle
# import streamlit as st
# import pandas as pd
# import numpy as np
#
# # Load the trained model
# with open('trained_model.pkl', 'rb') as file:
#     model = pickle.load(file)
#
# # Load the label encoder for categorical features
# with open('label_encoder.pkl', 'rb') as file:
#     label_encoder = pickle.load(file)
#
# # Load the standard scaler for numeric features
# with open('scaler.pkl', 'rb') as file:
#     scaler = pickle.load(file)
#
#
# # Streamlit app
# def main():
#     st.title('Loan Approval Predictor')
#
#     # Input form
#     st.sidebar.header('Input Features')
#
#     # Collect user input
#     user_input = {}
#     user_input['Gender'] = st.sidebar.radio('Gender', ['Male', 'Female'])
#     user_input['Married'] = st.sidebar.radio('Married', ['Yes', 'No'])
#     user_input['Dependents'] = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
#     user_input['Education'] = st.sidebar.radio('Education', ['Graduate', 'Not Graduate'])
#     user_input['Self_Employed'] = st.sidebar.radio('Self Employed', ['Yes', 'No'])
#     user_input['ApplicantIncome'] = st.sidebar.number_input('Applicant Income', value=0, step=1)
#     user_input['CoapplicantIncome'] = st.sidebar.number_input('Coapplicant Income', value=0, step=1)
#     user_input['LoanAmount'] = st.sidebar.number_input('Loan Amount', value=0, step=1)
#     user_input['Loan_Amount_Term'] = st.sidebar.number_input('Loan Amount Term', value=0, step=1)
#     user_input['Credit_History'] = st.sidebar.selectbox('Credit History', [0, 1])
#     user_input['Property_Area'] = st.sidebar.radio('Property Area', ['Rural', 'Semiurban', 'Urban'])
#
#     # Submit button
#     if st.sidebar.button('Submit'):
#         # Preprocess user input
#         input_df = pd.DataFrame([user_input])
#
#         # Transform the user input data using the label encoder fitted during training
#         input_df['Gender'] = label_encoder.transform(input_df['Gender'])
#         input_df['Married'] = label_encoder.transform(input_df['Married'])
#         input_df['Dependents'] = label_encoder.transform(input_df['Dependents'])
#         input_df['Education'] = label_encoder.transform(input_df['Education'])
#         input_df['Self_Employed'] = label_encoder.transform(input_df['Self_Employed'])
#         input_df['Credit_History'] = label_encoder.transform(input_df['Credit_History'])
#         input_df['Property_Area'] = label_encoder.transform(input_df['Property_Area'])
#
#         # Scale numeric features using standard scaler
#         numeric_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
#         input_df[numeric_features] = scaler.transform(input_df[numeric_features])
#
#         # Make prediction
#         prediction = model.predict(input_df)
#
#         # Display prediction
#         st.subheader('Prediction')
#         if prediction[0] == 'Y':
#             st.write('Congratulations! Your loan application is approved.')
#         else:
#             st.write('Sorry, your loan application is rejected.')
#
#
# if __name__ == '__main__':
#     main()




#========================================================================


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
        if prediction[0] == 'Y':
            st.write('Congratulations! Your loan application is approved.')
        else:
            st.write('Sorry, your loan application is rejected.')


if __name__ == '__main__':
    main()
