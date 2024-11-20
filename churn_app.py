import pandas as pd
import numpy as np 
import tensorflow as tf 
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow.keras.models import load_model

model = load_model(r'C:\Users\snehal.sankanna\AppData\Local\anaconda3\Churn_Mdelling\model.h5')

with open(r'C:\Users\snehal.sankanna\AppData\Local\anaconda3\Churn_Mdelling\label_encoder_gender.pkl', 'rb') as file:
    label_gender = pickle.load(file)

with open(r'C:\Users\snehal.sankanna\AppData\Local\anaconda3\Churn_Mdelling\onehotencoder_geo.pkl', 'rb') as file:
    label_geo = pickle.load(file)

with open(r'C:\Users\snehal.sankanna\AppData\Local\anaconda3\Churn_Mdelling\scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

#App
st.title('Customer Churn Prediction')
geography = st.selectbox('Geography', label_geo.categories_[0])
gender = st.selectbox('Gender', label_gender.classes_)
age = st.slider('Age', 18,92)
balence = st.number_input('Balence')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider("Tenure", 0,10)
num_of_products = st.number_input('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])


input_data = {
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [label_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure': [tenure],
    'Balance': [balence],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

# Geography Encoding
geo_encoded = label_geo.transform(np.array([[geography]]))  # Reshape input
geo_df = pd.DataFrame(
    geo_encoded.toarray(), 
    columns=label_geo.get_feature_names_out(['Geography'])
)

# Create the input data as a DataFrame
input_data = {
    'CreditScore': [credit_score],
    'Gender': [label_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balence],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
}

input_df = pd.DataFrame(input_data)

# Combine with geography data
input_combined = pd.concat([input_df.reset_index(drop=True), geo_df.reset_index(drop=True)], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_combined)

prediction = model.predict(input_scaled)

pred_prob = prediction[0][0]

st.write(f'Churn Probability: {pred_prob: .2f}')

if pred_prob>0.5:
    st.write("The person is likely to churn")
else:
    st.write("The person is not likely to churn")
