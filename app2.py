import streamlit as st 
import numpy as np 
import tensorflow as tf 
from sklearn.preprocessing import StandardScaler, LabelEncoder , OneHotEncoder
import pandas as pd 
import pickle 

# Load the trained model 
model = tf.keras.models.load_model('regression_model.h5')

#  Load the scaler , encoder 
with open('salary_gender_encoder.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('salary_geo_encoder.pkl','rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('salary_scaler.pkl','rb') as file:
    scaler = pickle.load(file)

#  Streamit app

st.title(' Salary Prediction - churn Regression Model')

# User Inputs
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender' , label_encoder_gender.classes_)
age = st.slider('Age', 18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

#  prepare input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
})

# One-hot encode Geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine input data with encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict churn probability
prediction = model.predict(input_scaled)
st.subheader('Predicted salary:')
st.write(f"Estimated Salary : {prediction[0][0]:.2f}")