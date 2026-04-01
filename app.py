import streamlit as st
import numpy as np
import pickle 
from sklearn.datasets import load_breast_cancer




# set app title and description

st.title("Breast Cancer Prediction App")
st.write("This app predicts whether a breast mass is malignant or benign based on the features of the mass.")


# load  saved model and scaler

with open('model.pkl', 'rb') as file:
    model=pickle.load( file)

with open('scaler.pkl', 'rb') as file:
    scaler=pickle.load( file)
    
    
    # data loading and preprocessing
    
data=load_breast_cancer()
features=data.feature_names

# create input fields for user to enter feature values
st.subheader("Enter the features of the breast mass:")
input_data={}
for feature in features:
    values=st.number_input(feature, value=0.0)
    
    input_data[feature]=values 
# convert input data to numpy array and reshape for prediction
input_array=np.array(list(input_data.values())).reshape(1, -1)
# scale the input data using the loaded scaler
scaled_input=scaler.transform(input_array)
# make prediction using the loaded model
if st.button("Predict"):
    prediction=model.predict(scaled_input)
    if prediction[0]==0:
        st.write("The breast mass is predicted to be benign.")
    else:
        st.write("The breast mass is predicted to be malignant.")
        
        
        
        