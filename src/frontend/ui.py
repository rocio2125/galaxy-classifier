import streamlit as st
import requests
import pandas as pd
from PIL import Image
import os

# Obtener la ruta absoluta del directorio donde está ESTE archivo (ui.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Rutas utilizadas en la aplicación
image_path_banner = os.path.join(current_dir, "assets", "galaxy.jpg")
image_path_metrics = os.path.join(current_dir, "assets", "metricas.png")
image_path_confusion = os.path.join(current_dir, "assets", "matriz_confusion.png")
music_path = os.path.join(current_dir, "assets", "sound.mp3")

# Leer variable de entorno, si no existe usa localhost (para pruebas sin docker)
API_URL = os.getenv("API_URL", "http://localhost:5000")

#API_URL = "https://galaxy-classifier-api.onrender.com" # Para pruebas en local



# --------------------------
# BANNER
# --------------------------
banner = Image.open(image_path_banner)
st.image(
    banner,
    use_container_width=True
)

# Título
st.markdown(
    """
    <h1 style="
        text-align: center;
        margin-top: -120px;     
        color: white;
        text-shadow: 3px 3px 8px black;
        font-size: 3.2rem;
    ">
        Clasificador de Galaxias
    </h1>
    """,
    unsafe_allow_html=True)

# --------------------------
# MÚSICA DE FONDO
# --------------------------

st.sidebar.header("🎵 Música de fondo")
try:
    with open(music_path, "rb") as f:
        audio_bytes = f.read()
    st.sidebar.audio(audio_bytes, format="audio/mp3")
except FileNotFoundError:
    st.sidebar.info("Archivo de música no encontrado. Cambia music_path o sube un archivo.")


# Crear pestañas
tab1, tab2, tab3 = st.tabs(["Predicción", "Sobre el proyecto","Resultados del modelo"])

with tab1:
    st.write("## 📡 Interfaz de predicción de galaxias")
    st.write(
        """
        Esta aplicación permite clasificar imágenes de galaxias. Sube una imagen de una galaxia y 
        el modelo predecirá su tipo y elnivel de confianza.
        """
    )
    # ====================================================================
    # 1️⃣ SUBIR IMAGEN Y PREDECIR
    # ====================================================================

    st.header("🔭 Realizar predicción")

    uploaded_file = st.file_uploader("Sube una imagen", type=["jpg", "png"])

    if st.button("Predecir"):
        if uploaded_file:
            # Mostrar imagen subida
            st.image(uploaded_file, caption="Imagen subida", width=300)

            uploaded_file.seek(0)
            bytes_data = uploaded_file.getvalue()
            files = {"image": (uploaded_file.name, bytes_data, uploaded_file.type)}

            resp = requests.post(f"{API_URL}/predict", files=files)

            if resp.status_code == 200:
                data = resp.json()
                st.success("Predicción exitosa")
                st.write(f"**Tipo de galaxia:** {data.get('prediction_name','fallo')}")
                st.write(f"**Confianza:** {data.get('confidence','?')}")
                st.write(f"**ID predicción:** {data.get('id')}")

            else:
                st.error(f"Error en API: {resp.text}")

        else:
            st.error("Primero sube una imagen.")


    st.write("---")


    # ====================================================================
    # 2️⃣ CONSULTAR TODAS LAS PREDICCIONES (TABLA)
    # ====================================================================

    st.header("📄 Ver todas las predicciones")

    if st.button("Ver predicciones"):
        resp = requests.get(f"{API_URL}/predictions")
        if resp.status_code == 200:
            data = resp.json()

            # Convertir a tabla
            df = pd.DataFrame(data)
            st.dataframe(df)
        else:
            st.error("No se pudieron obtener las predicciones")


    st.write("---")


    # ====================================================================
    # 3️⃣ CONSULTAR POR ID
    # ====================================================================

    st.header("🔎 Buscar predicción por ID")

    pred_id = st.number_input("ID:", min_value=1, step=1)

    if st.button("Buscar por ID"):
        resp = requests.get(f"{API_URL}/predictions/{pred_id}")
        st.write(resp.json())


    st.write("---")


    ##====================================================================
    ##4️⃣ CONSULTAR POR FILENAME
    ##====================================================================

    # st.header("📁 Buscar predicción por nombre de archivo")

    # filename = st.text_input("Nombre del archivo (ej: galaxy_01.png)")

    # if st.button("Buscar por filename"):
        # resp = requests.get(f"{API_URL}/predictions/filename/{filename}")
        # if resp.status_code == 200:
            # st.write(resp.json())
        # else:
            # st.error("Archivo no encontrado")


    # st.write("---")


    # ====================================================================
    # 5️⃣ RESET — BORRAR TODA LA BASE DE DATOS
    # ====================================================================

    st.header("🗑️ Resetear base de datos")

    if st.button("Borrar todas las predicciones"):
        resp = requests.delete(f"{API_URL}/predictions/reset")

        if resp.status_code == 200:
            st.success("Base de datos vaciada")
            st.write(resp.json())
        else:
            st.error("Error al borrar los datos")

with tab2:
    st.header("Objetivo del proyecto")

    texto_1 = """
    <p style="text-align: justify;">
    En la era del Big Data astronómico, la cantidad de información generada por los telescopios modernos 
    supera con creces la capacidad de realizar una clasificación manual eficiente. Dentro de este 
    escenario surge la necesidad de aplicar técnicas avanzadas de Inteligencia Artificial para 
    procesos que antes requerían años de trabajo humano. <br><br>
    La clasificación de galaxias es un área donde 
    esta transición resulta especialmente relevante, ya que la morfología galáctica permite comprender 
    la evolución y los procesos físicos internos que moldean estas estructuras cósmicas.<br><br>
    Nuestro proyecto se basa en el dataset Galaxy10 DECaLS, construido a partir de imágenes de alta 
    resolución del Dark Energy Camera Legacy Survey. Este conjunto de datos incluye alrededor de 
    18.000 galaxias categorizadas en diez clases morfológicas bien diferenciadas, lo que ofrece una 
    base sólida para entrenar modelos de aprendizaje automático capaces de reconocer patrones complejos 
    en las imágenes astronómicas.<br><br>
    La motivación tecnológica y de negocio detrás del proyecto responde al creciente interés en la 
    aplicación de IA dentro del sector astronómico, un campo en expansión pero con altos costos 
    operativos. <br><br>
    Automatizar tareas como la clasificación de galaxias reduce de manera significativa el tiempo
    y los recursos necesarios para procesar grandes volúmenes de datos. Además, abre oportunidades para colaborar con agencias espaciales, observatorios, institutos de investigación y consorcios de telescopios, quienes podrían beneficiarse de soluciones más rápidas, eficientes y escalables para el análisis del cielo profundo.
    En conjunto, este proyecto se posiciona en la intersección entre ciencia, tecnología y negocio, aprovechando el potencial de la inteligencia artificial para abordar uno de los desafíos más demandantes de la astronomía moderna.
    </p>
    """
    st.markdown(texto_1, unsafe_allow_html=True)

with tab3:
    st.header("Resultados del modelo")

    texto_2 = """
    <p style="text-align: justify;">
    Tras entrenar y evaluar nuestro modelo de clasificación de galaxias utilizando el dataset 
    Galaxy10 DECaLS, hemos obtenido resultados prometedores que demuestran la eficacia de las 
    técnicas de aprendizaje automático aplicadas. <br><br>
    El modelo alcanzó una accuracy del 79%, lo que indica su capacidad para
    identificar correctamente las diferentes clases morfológicas de galaxias.
    </p>
    """
    st.markdown(texto_2, unsafe_allow_html=True)

    st.header("Resumen de métricas")
    metrics = Image.open(image_path_metrics)
    st.image(
        metrics,
        use_container_width=False
    )

    st.header("Matriz de confusión")
    matrix = Image.open(image_path_confusion)
    st.image(
        matrix,
        use_container_width=False
    )
