import streamlit as st
import numpy as np
import pickle

pipe = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pickle.load(open('car.pkl', 'rb'))

st.title("Car-Price-Predictor")

company = st.selectbox('Select company', car['company'].unique())

model = st.selectbox('Select Model', car['name'].unique())

year = st.number_input('Year of Purchase')

fuel = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])

kms = st.number_input('Total kilometers travelled')

if st.button('Predict Price'):

    query = np.array([company, model, year, fuel, kms])
    query = query.reshape(1, 5)
    st.title("Predicted Price : " + pipe.predict(query)[0])




