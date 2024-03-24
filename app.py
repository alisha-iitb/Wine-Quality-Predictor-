import numpy as np
import pickle
import pandas as pd
import streamlit as st 
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image

#Load the pre-trained Random Forest Classifier
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)

# Define the main features
main_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                 'pH', 'sulphates', 'alcohol']

def welcome():
    return "Welcome All"

def main():
    st.title("White Wine Quality Predictor")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Bank Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
   
    # Initialize dictionary to store slider values
    slider_values = {}

    # Add sliders for input features
    for feature in main_features:
        default_value = 0.5  # Set default value for sliders
        min_value = 0.0      # Minimum value for sliders
        max_value = 1.0      # Maximum value for sliders
        step = 0.01          # Step size for sliders
        slider_label = feature.replace(' ', '_')  # Replace spaces with underscores for labels
        slider_value = st.slider(feature, min_value, max_value, default_value, step=step, key=slider_label)
        slider_values[feature] = slider_value
    
    # Preprocess input data
    input_data = preprocess_input_data(slider_values)
     
      

    # Make predictions
    prediction=classifier.predict(input_data)
    # Display the prediction
    st.write('Predicted Quality:', prediction[0])

def preprocess_input_data(slider_values):
    # Create a DataFrame with the input features
    input_data = pd.DataFrame(slider_values, index=[0])

    # Scaling the numerical features
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data)

    return input_data_scaled

if __name__=='__main__':
    main()
    
    
    