import streamlit as st
import pickle
import numpy as np

# Load the model and dataframe
#pipe = pickle.load(open('pipe.pkl','rb'))
#df = pickle.load(open('df.pkl','rb'))

pickle_in = open('pipe.pkl','rb')
pipe = pickle.load(pickle_in)

pickle_in01= open('df.pkl','rb')
df =pickle.load(pickle_in01)



st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand', df['Company'].unique())

# Type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# RAM
ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# Weight
weight = st.number_input('Weight of the Laptop (in kg)')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# Screen size
screen_size = st.slider('Screen size (in inches)', 10.0, 18.0, 13.0)

# Resolution
resolution = st.selectbox('Screen Resolution', [
    '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
    '2880x1800', '2560x1600', '2560x1440', '2304x1440'
])

# CPU
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

# HDD
hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

# GPU
gpu = st.selectbox('GPU', df['Gpu brand'].unique())

# OS
os = st.selectbox('OS', df['os'].unique())

# Prediction button
if st.button('Predict Price'):
    # Preprocess the input
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Prepare the query
    query = np.array([company, laptop_type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])

    # Reshape the query to match the model input
    query = query.reshape(1, 12)

    # Predict and display the result
    predicted_price = np.exp(pipe.predict(query)[0])  # Assuming the model predicts log-transformed prices
    st.title(f"The predicted price of this configuration is ₹{int(predicted_price)}")
