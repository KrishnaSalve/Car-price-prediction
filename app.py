import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle 


model = pickle.load(open('pipe.pkl', 'rb'))
data = pickle.load(open('car_data.pkl', 'rb'))


st.header("Car Price Predictor Model")


name = st.selectbox("Select Car Brand", data['name'].unique())

year = st.slider("Car Manufacturing Year", 1994, 2024)

km_driven = st.number_input("No. of Kms Driven")

fuel = st.selectbox("Fuel Type", data['fuel'].unique())

seller_type = st.selectbox("Seller Type", data['seller_type'].unique())

transmission = st.selectbox("Transmission Type", data['transmission'].unique())

owner = st.selectbox("Car Owner", data['owner'].unique())

mileage = st.slider("Car Mileage(in  kmpl)", np.min(data['mileage']), np.max(data['mileage']))

engine = st.slider("Car Engine(in  CC)", np.min(data['engine']), np.max(data['engine']))

max_power = st.slider("Car Max Power(in  bhp)", np.min(data['max_power']), np.max(data['max_power']))

seats = st.slider("No. of Seats", int(np.min(data['seats'])), int(np.max(data['seats'])))



if st.button("Predict"):
    input_data = pd.DataFrame([[name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, seats]], columns = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'seats'])


    car_price = model.predict(input_data)

    st.markdown("Predicted Car Price for given data would be Rs. " + str(np.round(np.exp(car_price[0]), 2)))