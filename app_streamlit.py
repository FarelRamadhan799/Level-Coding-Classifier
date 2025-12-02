import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model_level_coding_classifier.joblib")

st.title("Level Coding Classifier")
st.markdown("Model klasifikasi untuk memprediksi level coding siswa")

hours_coding_daily = st.slider("Hours Coding Daily", 1.0, 7.0, 3.0)
preferred_language = st.pills("Preferred Languange", ["Python", "C++", "Java"], default="Python")
typing_speed = st.slider("Typing Speed", 20, 100, 60)
import_usage = st.pills("Import Usage", ["Yes", "No"], default="Yes")
oop_usage = st.pills("Oop Usage", ["Yes", "No"], default="Yes")

if st.button("Prediksi", type="primary") :
	data = pd.DataFrame([[hours_coding_daily, preferred_language, typing_speed, import_usage, oop_usage]], columns=['hours_coding_daily', 'preferred_language', 'typing_speed', 'import_usage', 'oop_usage'])
	prediksi = model.predict(data)[0]
	presentase = max(model.predict_proba(data)[0])
	st.success(f"Prediksi : **{prediksi}** Presentase Keyakinan : **{presentase*100:.2f}%**")
	st.balloons()

st.divider()
st.caption("Dibuat oleh Farel Ramadhan")