import streamlit as st
import pickle
import numpy as np

# Load the saved Linear Regression model
with open('Automobile_displacement.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to predict EMISSION using the loaded model
def predict_displacement(cylinders, horsepower, weight):
    features = np.array([cylinders, horsepower, weight])
    features = features.reshape(1,-1)
    emission = model.predict(features)
    return emission[0]

# Streamlit UI
st.title('DISPLACEMMENT PREDICTION')
st.write("""
## AUTOMOBILES PREDICTION
ENTER THE VALUES FOR THE INPUT FEATURES TO PREDICT DISPLACEMENT.
""")

# Input fields for user
cylinders = st.number_input('CYLINDERS')
horsepower = st.number_input('HORSEPOWER')
weight = st.number_input('WEIGHT')

# Prediction button
if st.button('Predict'):
    # Predict EMISSION
    displacement_prediction = predict_displacement(cylinders, horsepower, weight)
    st.write(f"PREDICTED DISPLACEMENT: {displacement_prediction}")