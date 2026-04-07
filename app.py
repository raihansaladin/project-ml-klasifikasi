import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Klasifikasi Iklim Hybrid", layout="wide", page_icon="🌤️")

# 2. Fungsi Load Model (Cached agar ringan)
@st.cache_resource
def load_components():
    model  = joblib.load('model_xgboost.pkl')
    kmeans = joblib.load('kmeans_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le     = joblib.load('label_encoder.pkl')
    return model, kmeans, scaler, le

try:
    xgb_model, km_model, std_scaler, label_enc = load_components()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# 3. Fungsi Feature Engineering
# ⚠️  URUTAN dan NAMA fitur HARUS persis sama dengan notebook saat training
def add_features(df):
    df_fe = df.copy()
    df_fe['heat_index']         = df_fe['Temperature'] + (0.55 * df_fe['Humidity'])
    df_fe['uv_visibility']      = df_fe['UV Index'] / (df_fe['Cloud Cover'] + 1)
    df_fe['storm_factor']       = (1013 - df_fe['Atmospheric Pressure']) * df_fe['Wind Speed']
    df_fe['rain_fog_logic']     = df_fe['Precipitation (%)'] / (df_fe['Visibility (km)'] + 1)
    df_fe['pressure_deviation'] = abs(df_fe['Atmospheric Pressure'] - 1013)
    return df_fe

# Urutan kolom HARUS sama persis dengan X_train saat scaler.fit()
# Diambil langsung dari scaler.feature_names_in_
FEATURE_ORDER = [
    'Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover',
    'Atmospheric Pressure', 'UV Index', 'Season', 'Visibility (km)', 'Location',
    'heat_index', 'uv_visibility', 'storm_factor', 'rain_fog_logic', 'pressure_deviation'
]

# 4. Antarmuka Pengguna (UI)
st.title("🌤️ Sistem Klasifikasi Pola Iklim Regional")
st.markdown("### Pendekatan Hybrid: K-Means Clustering + XGBoost Classifier")
st.write("Gunakan form di bawah untuk memprediksi tipe cuaca berdasarkan parameter atmosfer.")
st.markdown("---")

col_input, col_info = st.columns([2, 1])

with col_input:
    st.subheader("📍 Input Data Cuaca")
    c1, c2 = st.columns(2)

    with c1:
        temp  = st.number_input("Temperature (°C)", value=25.0, step=0.5)
        hum   = st.number_input("Humidity (%)", value=60.0, min_value=0.0, max_value=100.0, step=1.0)
        wind  = st.number_input("Wind Speed (km/h)", value=10.0, min_value=0.0, step=0.5)
        prec  = st.number_input("Precipitation (%)", value=0.0, min_value=0.0, max_value=100.0, step=1.0)
        cloud = st.selectbox("Cloud Cover", [0, 1, 2, 3, 4],
                             format_func=lambda x: ["0 - Clear","1 - Partly Cloudy","2 - Cloudy","3 - Overcast","4 - Dense"][x])

    with c2:
        press  = st.number_input("Atmospheric Pressure (hPa)", value=1013.0, step=0.5)
        uv     = st.number_input("UV Index", value=5.0, min_value=0.0, max_value=11.0, step=0.5)
        vis    = st.number_input("Visibility (km)", value=10.0, min_value=0.0, step=0.5)
        # ⚠️ Season HARUS sebelum Location (sesuai urutan fit)
        season = st.selectbox("Season (Musim)", [0, 1, 2, 3],
                              format_func=lambda x: ["Winter","Spring","Summer","Autumn"][x])
        loc    = st.selectbox("Location Type", [0, 1, 2],
                              format_func=lambda x: ["Inland (Daratan)","Mountain (Pegunungan)","Coastal (Pesisir)"][x])

with col_info:
    st.subheader("📖 Keterangan Parameter")
    with st.expander("Lihat Penjelasan", expanded=True):
        st.write("""
        - **Temperature**: Suhu udara (°C)
        - **Humidity**: Kelembapan udara (%)
        - **Wind Speed**: Kecepatan angin (km/jam)
        - **Precipitation**: Curah hujan / potensi hujan (%)
        - **Cloud Cover**: Tingkat tutupan awan (0=Clear, 4=Dense)
        - **Pressure**: Tekanan udara (standar: 1013 hPa)
        - **UV Index**: Intensitas radiasi ultraviolet (0-11)
        - **Visibility**: Jarak pandang maksimal (km)
        - **Season**: Musim saat pengukuran
        - **Location**: Tipe lokasi geografis
        """)
    st.subheader("🎯 Kelas Output")
    st.info("Cloudy | Rainy | Snowy | Sunny")

# 5. Logika Prediksi
st.markdown("---")
if st.button("🚀 Jalankan Prediksi Klasifikasi", use_container_width=True):
    try:
        # A. Buat DataFrame input mentah
        #    ⚠️ Season dan Location sudah dalam urutan yang sesuai FEATURE_ORDER
        input_raw = pd.DataFrame([{
            'Temperature'         : temp,
            'Humidity'            : hum,
            'Wind Speed'          : wind,
            'Precipitation (%)'   : prec,
            'Cloud Cover'         : cloud,
            'Atmospheric Pressure': press,
            'UV Index'            : uv,
            'Season'              : season,   # Season DULU
            'Visibility (km)'     : vis,
            'Location'            : loc,      # baru Location
        }])

        # B. Feature Engineering (nama & urutan harus sama dengan notebook)
        df_fe = add_features(input_raw)

        # C. Susun kolom sesuai FEATURE_ORDER (kunci anti-error)
        df_ready = df_fe[FEATURE_ORDER]

        # D. Scaling
        input_scaled = std_scaler.transform(df_ready)

        # E. Prediksi Cluster (Hybrid Step)
        cluster_label = km_model.predict(input_scaled)

        # F. Gabungkan fitur + cluster_label untuk XGBoost
        input_final = np.append(input_scaled, [[cluster_label[0]]], axis=1)

        # G. Prediksi akhir
        prediction_idx = xgb_model.predict(input_final)
        result_text    = label_enc.inverse_transform(prediction_idx)[0]

        # H. Tampilkan hasil
        emoji_map = {'Sunny':'☀️','Rainy':'🌧️','Cloudy':'☁️','Snowy':'❄️'}
        emoji = emoji_map.get(result_text, '🌤️')

        st.balloons()
        st.markdown(f"""
        <div style="background-color:#d4edda; padding:25px; border-radius:12px;
                    border:2px solid #28a745; text-align:center;">
            <h1 style="color:#155724; margin:0;">{emoji} {result_text}</h1>
            <p style="color:#155724; font-size:16px; margin-top:8px;">
                Kategori Cluster AI: <b>{cluster_label[0]}</b>
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Detail probabilitas
        if hasattr(xgb_model, 'predict_proba'):
            proba = xgb_model.predict_proba(input_final)[0]
            classes = label_enc.classes_
            st.markdown("#### 📊 Probabilitas per Kelas")
            prob_df = pd.DataFrame({'Kelas': classes, 'Probabilitas': proba})\
                        .sort_values('Probabilitas', ascending=False)
            for _, row in prob_df.iterrows():
                st.progress(float(row['Probabilitas']),
                            text=f"{row['Kelas']}: {row['Probabilitas']*100:.1f}%")

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")
        st.exception(e)

st.markdown("---")
st.caption("Developed for University Project – Informatics Engineering | Machine Learning Class C")