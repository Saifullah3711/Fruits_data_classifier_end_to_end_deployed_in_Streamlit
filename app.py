# Importing required Libraries
import streamlit as st
from joblib import load
import numpy as np
from sklearn.preprocessing import StandardScaler

# Scaler for Scaling the Data -> Input data must be scaled
# because the model is trained on scaled data
scaler = StandardScaler()

st.title("Fruit Data Classifier")

# Function to Take input from users and return scaled numpy array
def user_data():
    mass = st.number_input(label="mass", step=1.0, format="%.3f")
    width = st.number_input(label="width", step=1.0, format="%.3f")
    height = st.number_input(label="height", step=1.0, format="%.3f")
    color_score = st.number_input(label="color_score", step=1.0, format="%.3f")
    fruit_features = np.array([[mass, width, height, color_score]])
    fruit_features_scaled = scaler.fit_transform(fruit_features)
    return fruit_features_scaled


data_for_model = user_data()

# Loading the Saved Model
model = load("Fruits_classifier.joblib")

press_button = st.button("Predict")
if press_button:
    result = model.predict(data_for_model)
   # Dictionary for the key values
   # The Model will predict values from 1 - 4 
    fruit_label = {
        1 : "Apple",
        2: "Mandarin",
        3: "Orange",
        4: "Lemon"
    }
    
    st.title("The Predicted Fruit is {}" .format(fruit_label[result[0]]))