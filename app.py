import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load semua file yang sudah disimpan
model = joblib.load('model_xgboost.pkl')
scaler = joblib.load('scaler.pkl')
le = joblib.load('label_encoder.pkl')
kmeans = joblib.load('kmeans_model.pkl')

st.title("Aplikasi Klasifikasi Pola Iklim Regional")
st.write("Masukkan data cuaca di bawah ini untuk memprediksi tipe iklim.")

# Buat Input Form (Sesuaikan dengan kolom datasetmu)
temp = st.number_input("Temperature (C)", value=25.0)
hum = st.number_input("Humidity (%)", value=60.0)
wind = st.number_input("Wind Speed", value=10.0)
prec = st.number_input("Precipitation (%)", value=0.0)
cloud = st.selectbox("Cloud Cover", [0, 1, 2, 3, 4]) # Contoh kategori

if st.button("Prediksi"):
    # 1. Gabungkan input menjadi dataframe
    input_data = pd.DataFrame([[temp, hum, wind, prec, cloud]], 
                              columns=['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover'])
    
    # 2. Tambahkan Fitur Baru (Samakan dengan fungsi add_features di notebook)
    input_data['heat_index'] = input_data['Temperature'] + (0.55 * input_data['Humidity'])
    # ... tambahkan fitur lainnya sesuai notebook ...

    # 3. Scaling & Predict
    input_scaled = scaler.transform(input_data)
    
    # Tambahkan cluster label jika pakai Hybrid
    cluster = kmeans.predict(input_scaled)
    input_final = np.append(input_scaled, [[cluster[0]]], axis=1)
    
    prediction = model.predict(input_final)
    result = le.inverse_transform(prediction)

    st.success(f"Hasil Prediksi Tipe Cuaca: **{result[0]}**")