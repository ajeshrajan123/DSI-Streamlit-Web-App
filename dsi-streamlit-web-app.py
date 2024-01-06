# -*- coding: utf-8 -*-
"""
Created on Sat Jan  6 14:03:50 2024

@author: ajesh
"""

import streamlit as st
import pandas as pd
import joblib

# load our model pipeline object

model = joblib.load("model.joblib")

# add title and instructions

st.title("Purchase Prediction Model")
st.subheader("Enter customer information and submit for likelihood to purchase")

#age input form
age = st.number_input(
    label = "01. Enter the Customer's Age",
    min_value=18,
    max_value=120,
    value = 35)


#gender input form
gender = st.radio(label = "02. Select the Customer's Gender", 
                  options = ['M','F'])

#credit score input form
credit_score = st.number_input(
    label = "03. Enter Customer's Credit Score",
    min_value=0,
    max_value=1000,
    value = 500)

#submit inputs to model

if st.button("Submit for Prediction"):
    # store our date in a dataframe for prediction
    
    new_data = pd.DataFrame ({"age": [age],
                              "gender":[gender],
                              "credit_score":[credit_score]})
    
    # apply model pipeline to the input the data & extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction
    st.subheader(f"Based on the customer's attributes, our model predicts a purchase probaility of {pred_proba:.0%}")
    
    
    
