import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Iklim Hybrid", layout="wide", page_icon="🌤️")

# 2. Fungsi Load Model (Cached agar ringan)
@st.cache_resource
def load_components():
    model  = joblib.load('model_best_90_10.pkl')
    kmeans = joblib.load('kmeans_model_90_10.pkl')
    scaler = joblib.load('scaler_90_10.pkl')
    le     = joblib.load('label_encoder.pkl')
    return model, kmeans, scaler, le

try:
    rf_model, km_model, std_scaler, label_enc = load_components()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 3. Fungsi Feature Engineering
def add_features(df):
    df_fe = df.copy()
    df_fe['heat_index']         = df_fe['Temperature'] + (0.1 * df_fe['Humidity'])
    df_fe['uv_visibility']      = df_fe['UV Index'] / (df_fe['Cloud Cover'] + 1)
    df_fe['storm_factor']       = (1013 - df_fe['Atmospheric Pressure']) * df_fe['Wind Speed']
    df_fe['rain_fog_logic']     = df_fe['Precipitation (%)'] / (df_fe['Visibility (km)'] + 1)
    df_fe['pressure_deviation'] = abs(df_fe['Atmospheric Pressure'] - 1013)
    return df_fe

FEATURE_ORDER = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover',
    'Atmospheric Pressure', 'UV Index', 'Season', 'Visibility (km)', 'Location',
    'heat_index', 'uv_visibility', 'storm_factor', 'rain_fog_logic', 'pressure_deviation'
]

# 4. Antarmuka Pengguna (UI)
st.title("🌤️ Sistem Klasifikasi Pola Iklim Regional")
st.markdown("### Pendekatan Hybrid: K-Means Clustering + Random Forest")
st.write("Gunakan form di bawah untuk memprediksi tipe cuaca.")

col_input, col_info = st.columns([2, 1])

with col_input:
    st.subheader("📍 Input Data Cuaca")
    c1, c2 = st.columns(2)

    with c1:
        temp  = st.number_input("Temperature (°C)", value=25.0, step=0.5)
        hum   = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0)
        wind  = st.number_input("Wind Speed (km/h)", value=10.0, min_value=0.0)
        prec  = st.number_input("Precipitation (%)", value=0.0, min_value=0.0, max_value=100.0)
        cloud = st.selectbox("Cloud Cover", [0, 1, 2, 3, 4],
                             format_func=lambda x: ["0 - Clear","1 - Partly Cloudy","2 - Cloudy","3 - Overcast","4 - Dense"][x])

    with c2:
        press  = st.number_input("Atmospheric Pressure (hPa)", value=1013.0)
        uv     = st.number_input("UV Index", value=5.0, min_value=0.0, max_value=11.0)
        vis    = st.number_input("Visibility (km)", value=10.0, min_value=0.0)
        season = st.selectbox("Season (Musim)", [0, 1, 2, 3],
                              format_func=lambda x: ["Winter","Spring","Summer","Autumn"][x])
        loc    = st.selectbox("Location Type", [0, 1, 2],
                              format_func=lambda x: ["Inland","Mountain","Coastal"][x])

with col_info:
    st.subheader("🎯 Prediksi Target")
    st.markdown("""
    Agar model dapat memprediksi masing-masing kelas, coba gunakan kombinasi parameter berikut:
    
    * **Sunny**: Temp > 25°C, Hum < 40%, Prec 0%, Cloud 0.
    * **Rainy**: Hum > 80%, Prec > 70%, Cloud 3-4.
    * **Cloudy**: Hum 50-70%, Prec < 20%, Cloud 2-3.
    * **Snowy**: Temp < 0°C, Prec > 50%, Hum > 70%.
    """)

# 5. Logika Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Prediksi Klasifikasi", use_container_width=True):
    try:
        input_raw = pd.DataFrame([{
            'Temperature': temp, 'Humidity': hum, 'Wind Speed': wind,
            'Precipitation (%)': prec, 'Cloud Cover': cloud,
            'Atmospheric Pressure': press, 'UV Index': uv,
            'Season': season, 'Visibility (km)': vis, 'Location': loc,
        }])

        df_fe = add_features(input_raw)
        df_ready = df_fe[FEATURE_ORDER]
        input_scaled = std_scaler.transform(df_ready)

        # Prediksi Cluster (Hybrid)
        cluster_label = km_model.predict(input_scaled)

        # Gabungkan Fitur + Cluster untuk Random Forest
        input_final = np.append(input_scaled, [[cluster_label[0]]], axis=1)

        # Prediksi Akhir
        prediction_idx = rf_model.predict(input_final)
        result_text    = label_enc.inverse_transform(prediction_idx)[0]

        # Tampilkan hasil
        emoji_map = {'Sunny':'☀️','Rainy':'🌧️','Cloudy':'☁️','Snowy':'❄️'}
        st.balloons()
        st.markdown(f"""
            <div style="background-color:#2F7D9F; padding:20px; border-radius:10px; text-align:center;">
                <h1 style="margin:0;">{emoji_map.get(result_text, '🌤️')} {result_text}</h1>
                <p>Cluster: {cluster_label[0]}</p>
            </div>
        """, unsafe_allow_html=True)

        # Probabilitas per kelas
        if hasattr(rf_model, 'predict_proba'):
            proba = rf_model.predict_proba(input_final)[0]
            st.write("#### 📊 Probabilitas:")
            for i, p in enumerate(proba):
                st.write(f"{label_enc.classes_[i]}: {p*100:.1f}%")
                st.progress(float(p))

    except Exception as e:
        st.error(f"❌ Error: {e}")

st.markdown("---")
st.caption("Developed for University Project – Informatics Engineering | Machine Learning Class C")