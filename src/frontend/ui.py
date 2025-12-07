import streamlit as st
import requests
import os

# Leer variable de entorno, si no existe usa localhost (para pruebas sin docker)
API_URL = os.getenv("API_URL", "http://localhost:5000")

st.title("Cliente Streamlit para modelo de imágenes")


# SUBIR IMAGEN Y PREDECIR

uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

if st.button("Predecir"):
    if uploaded_file:
        # --- Imprimir qué llega ---
        print(f"📸 API Recibió archivo: {uploaded_file.name}")

        # Rebobina el archivo al principio
        uploaded_file.seek(0)

        # Leer bytes
        bytes_data = uploaded_file.getvalue()

        # Enviar
        files = {"image": (uploaded_file.name, bytes_data, uploaded_file.type)}
        resp = requests.post(f"{API_URL}/predict", files=files)

        if resp.status_code == 200:
            st.success("Predicción exitosa")
            st.write(resp.json())
        else:
            st.error(f"Error en API: {resp.text}")
    else:
        st.error("Primero sube una imagen.")


# CONSULTAR TODAS LAS PREDICCIONES

if st.button("Consultar todas las predicciones"):
    resp = requests.get(f"{API_URL}/predictions")
    st.write(resp.json())


# CONSULTAR UNA PREDICCIÓN POR ID

st.subheader("Consultar predicción por ID")
pred_id = st.number_input("ID:", min_value=1, step=1)

if st.button("Consultar por ID"):
    resp = requests.get(f"{API_URL}/predictions/{pred_id}")
    st.write(resp.json())


# BORRAR TODAS LAS PREDICCIONES

if st.button("Borrar todas las predicciones"):
    resp = requests.delete(f"{API_URL}/predictions/delete")
    st.success("Tabla de predicciones borrada")
    st.write(resp.json())