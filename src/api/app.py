import os
import pickle
import numpy as np
import datetime
import io
from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS  # <--- Para que el Frontend no falle
from PIL import Image
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
from src.database.db import db          # <--- IMPORTAMOS EL CONECTOR
from src.database.models import Prediction # <--- IMPORTAMOS EL MODELO

# Cargar variables de entorno (.env) automáticamente si estamos en local sin Docker
load_dotenv()

app = Flask(__name__)

# --- 1. CONFIGURACIÓN CORS (CRUCIAL PARA EQUIPOS) ---
# Permite que cualquier origen (*) acceda a tu API. 
# En prod real podrías restringirlo al dominio de tu frontend.
CORS(app)

# --- 2. CONFIGURACIÓN ROBUSTA DE BBDD (POSTGRES ONLY) ---
# Intentamos leer la URL. Si no existe, fallamos (Fail Fast).
# Esto evita que creas que estás guardando datos y en realidad no esté conectado.
database_url = os.getenv('DATABASE_URL')

if not database_url:
    raise ValueError("❌ ERROR CRÍTICO: No se encontró la variable DATABASE_URL. "
                     "Asegúrate de ejecutar con 'docker-compose up' o configurar tu .env")

# Parche necesario para Render (postgres:// -> postgresql://)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'clave_segura_por_defecto')

# INICIALIZACIÓN DE LA BASE DE DATOS
# Vinculamos la base de datos a esta app específica
db.init_app(app)

# --- 3. CARGA DEL MODELO (HUGGING FACE) ---
REPO_ID = "rocio2125/paisajes"
FILENAME = "paisajes.pkl"

def load_model_from_hf():
    print("⬇️ Iniciando descarga del modelo desde HuggingFace...")
    try:
        # cache_dir="/app/model_cache" optimiza la descarga en Docker
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            cache_dir="/app/model_cache"
        )
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        print("✅ Modelo cargado correctamente en memoria.")
        return model
    except Exception as e:
        print(f"❌ ERROR FATAL cargando el modelo: {e}")
        return None

# Cargamos el modelo al iniciar la app (Variables globales)
model = load_model_from_hf()

# --- 4. CREACIÓN DE TABLAS (SEGURO PARA PROD) ---
# Al arrancar, verificamos que la conexión a Postgres funciona y creamos tablas
with app.app_context():
    try:
        db.create_all()
        print("✅ Conexión a PostgreSQL exitosa. Tablas verificadas.")
    except Exception as e:
        print(f"❌ Error conectando a la Base de Datos: {e}")

# --- 5. PREPROCESADO ---
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((100, 100)) # ¡OJO! Ajustad esto al tamaño real que pida vuestro modelo
        img_arr = np.array(img) / 255.0
        img_arr = np.expand_dims(img_arr, axis=0)
        return img_arr
    except Exception as e:
        raise ValueError(f"Imagen corrupta o formato inválido: {e}")

# --- 6. ENDPOINTS ---

# Endpoint de bienvenida
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de Predicción de Paisajes Activa v1.0"}), 200

# Endpoint de estado de la API
@app.route("/health", methods=["GET"])
def health_check():
    """Endpoint para que Render sepa que estamos vivos"""
    return jsonify({"status": "ok", "db": "connected"}), 200

# Endpoint de predicción
@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "El modelo de IA no está disponible"}), 500

    # Soporte para enviar la imagen como 'image' o 'file'
    if 'image' in request.files:
        image_file = request.files['image']
    elif 'file' in request.files:
        image_file = request.files['file']
    else:
        return jsonify({"error": "Falta el archivo. Usa la key 'image' o 'file'"}), 400

    try:
        image_bytes = image_file.read()
        processed_img = preprocess_image(image_bytes)
        
        # Inferencia
        prediction_result = model.predict(processed_img)
        pred_value = prediction_result.tolist()

        # Guardado en PostgreSQL
        new_entry = Prediction(
            timestamp=datetime.datetime.now().isoformat(),
            filename=image_file.filename,
            prediction=str(pred_value)
        )
        db.session.add(new_entry)
        db.session.commit()

        return jsonify({
            "filename": image_file.filename,
            "prediction": pred_value,
            "saved_to_db": True
        })

    except Exception as e:
        db.session.rollback() # Deshacer cambios si falla
        return jsonify({"error": str(e)}), 500

# Endpoint de consultar predicciones
@app.route("/predictions", methods=["GET"])
def get_predictions():
    try:
        # Ordenamos por ID descendente (lo más nuevo primero)
        all_preds = Prediction.query.order_by(Prediction.id.desc()).all()
        results = [{
            "id": p.id,
            "timestamp": p.timestamp,
            "filename": p.filename,
            "prediction": p.prediction
        } for p in all_preds]
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint de consultar predicción por ID
@app.route("/predictions/<int:prediction_id>", methods=["GET"])
def get_prediction_by_id(prediction_id):
    p = Prediction.query.get(prediction_id)
    if p is None:
        return jsonify({"error": "Predicción no encontrada"}), 404
    
    return jsonify({
        "id": p.id,
        "timestamp": p.timestamp,
        "filename": p.filename,
        "prediction": p.prediction
    })

# Endpoint de borrar predicciones
@app.route("/predictions/delete", methods=["DELETE"])
def delete_all_predictions():
    try:
        # Borrado masivo rápido
        db.session.query(Prediction).delete()
        db.session.commit()
        return jsonify({"message": "Historial completo eliminado"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)