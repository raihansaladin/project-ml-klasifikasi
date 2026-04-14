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
    # Pastikan label_enc memiliki 4 kelas
    expected_classes = ['Cloudy', 'Rainy', 'Snowy', 'Sunny']
    if not all(c in label_enc.classes_ for c in expected_classes):
        st.warning("⚠️ Model mungkin tidak mendukung semua 4 kelas cuaca")
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 3. Fungsi Feature Engineering (IDENTIK dengan notebook)
def add_features(df):
    df_fe = df.copy()
    df_fe['heat_index']         = df_fe['Temperature'] + (0.1 * df_fe['Humidity'])
    df_fe['uv_visibility']      = df_fe['UV Index'] / (df_fe['Cloud Cover'] + 1)
    df_fe['storm_factor']       = (1013 - df_fe['Atmospheric Pressure']) * df_fe['Wind Speed']
    df_fe['rain_fog_logic']     = df_fe['Precipitation (%)'] / (df_fe['Visibility (km)'] + 1)
    df_fe['pressure_deviation'] = abs(df_fe['Atmospheric Pressure'] - 1013)
    return df_fe

# Urutan fitur harus SAMA PERSIS dengan training
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
        temp  = st.number_input("🌡️ Temperature (°C)", value=25.0, step=0.5)
        hum   = st.number_input("💧 Humidity (%)", value=60.0, min_value=0.0, max_value=100.0)
        wind  = st.number_input("💨 Wind Speed (km/h)", value=10.0, min_value=0.0)
        prec  = st.number_input("☔ Precipitation (%)", value=0.0, min_value=0.0, max_value=100.0)
        cloud = st.selectbox("☁️ Cloud Cover", [0, 1, 2, 3, 4],
                             format_func=lambda x: ["0 - Clear","1 - Partly Cloudy","2 - Cloudy","3 - Overcast","4 - Dense"][x])

    with c2:
        press  = st.number_input("📊 Atmospheric Pressure (hPa)", value=1013.0)
        uv     = st.number_input("☀️ UV Index", value=5.0, min_value=0.0, max_value=14.0)
        vis    = st.number_input("👁️ Visibility (km)", value=10.0, min_value=0.0)
        season = st.selectbox("🍂 Season (Musim)", [0, 1, 2, 3],
                              format_func=lambda x: ["Winter","Spring","Summer","Autumn"][x])
        loc    = st.selectbox("📍 Location Type", [0, 1, 2],
                              format_func=lambda x: ["Inland","Mountain","Coastal"][x])

with col_info:
    st.subheader("🎯 Tips Mencapai 4 Kelas Cuaca")
    st.markdown("""
    Untuk mendapatkan semua kelas target (**Cloudy, Rainy, Snowy, Sunny**), gunakan kombinasi berikut:
    
    ---
    **☀️ SUNNY** (Cerah)
    - Temp > 25°C
    - Humidity < 50%
    - Precipitation < 10%
    - Cloud Cover: 0-1
    - UV Index > 6
    
    **☁️ CLOUDY** (Berawan)
    - Humidity 50-80%
    - Cloud Cover: 2-3
    - Precipitation < 30%
    
    **🌧️ RAINY** (Hujan)
    - Humidity > 80%
    - Precipitation > 60%
    - Cloud Cover: 3-4
    
    **❄️ SNOWY** (Salju)
    - Temperature < 3°C
    - Precipitation > 50%
    - Humidity > 75%
    ---
    
    💡 **Hint**: Kombinasi ekstrim akan menghasilkan kelas yang sesuai
    """)

# 5. Logika Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Prediksi Klasifikasi", use_container_width=True):
    try:
        # Buat DataFrame input
        input_raw = pd.DataFrame([{
            'Temperature': temp, 'Humidity': hum, 'Wind Speed': wind,
            'Precipitation (%)': prec, 'Cloud Cover': cloud,
            'Atmospheric Pressure': press, 'UV Index': uv,
            'Season': season, 'Visibility (km)': vis, 'Location': loc,
        }])

        # Lakukan feature engineering
        df_fe = add_features(input_raw)
        
        # Pastikan semua kolom yang diperlukan ada
        missing_cols = set(FEATURE_ORDER) - set(df_fe.columns)
        if missing_cols:
            st.error(f"❌ Fitur yang hilang: {missing_cols}")
            st.stop()
        
        # Urutkan kolom sesuai FEATURE_ORDER
        df_ready = df_fe[FEATURE_ORDER]
        
        # Standard scaling
        input_scaled = std_scaler.transform(df_ready)

        # Prediksi Cluster (Hybrid)
        cluster_label = km_model.predict(input_scaled)

        # Gabungkan Fitur + Cluster untuk Random Forest
        import numpy as np
        input_final = np.append(input_scaled, [[cluster_label[0]]], axis=1)

        # Prediksi Akhir
        prediction_idx = rf_model.predict(input_final)
        result_text    = label_enc.inverse_transform(prediction_idx)[0]

        # Emoji mapping untuk 4 kelas
        emoji_map = {
            'Sunny': '☀️', 
            'Rainy': '🌧️', 
            'Cloudy': '☁️', 
            'Snowy': '❄️'
        }
        
        # Warna background sesuai kelas
        color_map = {
            'Sunny': '#F39C12',
            'Rainy': '#2980B9', 
            'Cloudy': '#7F8C8D',
            'Snowy': '#5DADE2'
        }
        bg_color = color_map.get(result_text, '#2F7D9F')
        
        st.balloons()
        st.markdown(f"""
            <div style="background-color:{bg_color}; padding:25px; border-radius:15px; text-align:center;">
                <h1 style="margin:0; font-size:3em;">{emoji_map.get(result_text, '🌤️')} {result_text}</h1>
                <p style="margin:10px 0 0 0; opacity:0.9;">Cluster: {cluster_label[0]} | Confidence: high</p>
            </div>
        """, unsafe_allow_html=True)

        # Probabilitas per kelas (4 kelas)
        if hasattr(rf_model, 'predict_proba'):
            proba = rf_model.predict_proba(input_final)[0]
            st.write("#### 📊 Distribusi Probabilitas:")
            
            # Buat DataFrame untuk probabilitas
            prob_df = pd.DataFrame({
                'Kelas Cuaca': label_enc.classes_,
                'Probabilitas': proba * 100
            }).sort_values('Probabilitas', ascending=False)
            
            # Tampilkan bar chart dengan warna
            colors = ['#e74c3c' if p == prob_df['Probabilitas'].max() else '#3498db' 
                     for p in prob_df['Probabilitas']]
            
            st.dataframe(
                prob_df.style.format({'Probabilitas': '{:.1f}%'}).background_gradient(cmap='Blues', subset=['Probabilitas']),
                use_container_width=True
            )
            
            # Progress bar untuk kelas tertinggi
            top_class = prob_df.iloc[0]['Kelas Cuaca']
            top_prob = prob_df.iloc[0]['Probabilitas']
            st.markdown(f"**🎯 Prediksi Utama: {top_class}**")
            st.progress(top_prob / 100)

    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Pastikan semua input terisi dengan benar dan model telah dilatih dengan data yang sesuai.")

st.markdown("---")
st.caption("Developed for University Project – Informatics Engineering | Machine Learning Class C")