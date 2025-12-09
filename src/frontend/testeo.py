import streamlit as st
import requests

API_URL = "http://127.0.0.1:5000"

st.title("Clasificador de Galaxias 🌌")

# =============================
# PREDICCIÓN
# =============================
image = st.file_uploader("Selecciona imagen", type=["jpg", "jpeg", "png"])

if st.button("Realizar predicción"):
    if image is None:
        st.error("Debes seleccionar una imagen.")
    else:
        files = {"image": (image.name, image.getvalue(), image.type)}
        resp = requests.post(f"{API_URL}/predict", files=files)

        if resp.status_code == 200:
            data = resp.json()
            st.success("Predicción generada:")
            st.write(f"**ID:** {data['id']}")
            st.write(f"**Fecha:** {data['timestamp']}")
            st.write(f"**Archivo:** {data['filename']}")
            st.write(f"**Predicción:** `{data['prediction_name']}`")
        else:
            st.error("Error en la predicción")

# =============================
# VER TODAS
# =============================
st.subheader("📄 Ver todas las predicciones")

if st.button("Cargar predicciones"):
    resp = requests.get(f"{API_URL}/predictions")
    if resp.status_code == 200:
        st.write(resp.json())
    else:
        st.error("No se pudieron cargar")

# =============================
# CONSULTAR POR ID
# =============================
st.subheader("🔍 Buscar predicción por ID")

pred_id = st.number_input("ID:", min_value=1, step=1)

if st.button("Buscar"):
    resp = requests.get(f"{API_URL}/predictions/{int(pred_id)}")
    if resp.status_code == 200:
        st.write(resp.json())
    else:
        st.error("ID no encontrado")

# =============================
# BORRAR TODO
# =============================
st.subheader("🗑️ Borrar todas las predicciones")

if st.button("Eliminar todo"):
    resp = requests.delete(f"{API_URL}/predictions/delete")
    if resp.status_code == 200:
        st.success(resp.json()["message"])
    else:
        st.error("No se pudo borrar")
        
# ===========================
#   BOTÓN PARA RESETEAR DB
# ===========================

st.subheader("⚠️ Administración de la Base de Datos")

if st.button("🗑️ Resetear Base de Datos"):
    try:
        resp = requests.post(f"{API_URL}/reset_db")
        if resp.status_code == 200:
            st.success("✅ Base de datos reseteada correctamente.")
            st.write(resp.json())
        else:
            st.error(f"❌ Error al resetear: {resp.status_code}")
            st.write(resp.text)
    except Exception as e:
        st.error("❌ No se pudo conectar con la API.")
        st.write(str(e))