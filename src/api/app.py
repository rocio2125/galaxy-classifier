import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

# --- IMPORTS DEL MODELO DE "GALAXIAS" (TENSORFLOW) ---
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download

# --- IMPORTS ---
from src.database.db import db
from src.database.models import Prediction
import datetime

# Cargar variables de entorno (.env) automáticamente si estamos en local sin Docker
load_dotenv()

app = Flask(__name__)

# 1. CONFIGURACIÓN CORS
# Permite que cualquier origen acceda a tu API. 
CORS(app)

# 2. CONFIGURACIÓN DE BASE DE DATOS
database_url = os.getenv('DATABASE_URL')

if not database_url:
    raise ValueError("❌ ERROR: No se encontró la variable DATABASE_URL. "
                     "Asegúrate de ejecutar con 'docker-compose up' o configurar tu .env")

# Parche para Render (postgres:// -> postgresql://)
if database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

app.config['SQLALCHEMY_DATABASE_URI'] = database_url
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'super-secret-key')

# Inicializamos la DB con la app
db.init_app(app)

# 3. CARGA DEL MODELO DE GALAXIAS (TENSORFLOW + HUGGING FACE)
# Usamos el repo pero con caché persistente
REPO_ID = "jprizzi/galaxy-classifier" # Poner el repo en HF
FILENAME = "modelo_galaxias.keras" # Asegúrate que este sea el nombre exacto en HF
CACHE_DIR = "/app/model_cache"     # Coincide con docker-compose.yml

print("⏳ Iniciando carga del modelo de Galaxias...")

try:
    model_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=FILENAME,
        cache_dir=CACHE_DIR
    )
    print(f"✅ Modelo descargado/encontrado en: {model_path}")

    # Cargamos el modelo de Keras
    model = load_model(model_path)
    print("🚀 Modelo de Galaxias cargado en memoria!")
except Exception as e:
    print(f"❌ ERROR CRÍTICO cargando el modelo: {e}")
    model = None

# Etiquetas del modelo de Galaxias (Mapeo de índice a nombre)
LABELS = {
    0: "spiral", 1: "elliptical", 2: "lenticular", 3: "irregular",
    4: "merger", 5: "unknown", 6: "barred spiral", 7: "compact",
    8: "edge_on", 9: "other"
}

# 4. CREACIÓN DE TABLAS
# Al arrancar, verificamos que la conexión a Postgres funciona y creamos tablas
with app.app_context():
    try:
        db.create_all()
        print("✅ Conexión a PostgreSQL exitosa. Tablas verificadas.")
    except Exception as e:
        print(f"❌ Error conectando a la Base de Datos: {e}")


# 5. ENDPOINTS

# ENDPOINT: BIENVENIDA
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API de Predicción de Galaxias Activa v1.0"}), 200

# ENDPOINT: ESTADO DE LA API
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "model": "galaxy-classifier-v1"}), 200


# PREPROCESADO
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        arr = np.array(img) / 255.0

        # LOG DE DIAGNÓSTICO
        print(f"🔍 [DEBUG IMG] Shape: {arr.shape} | Min: {arr.min():.4f} | Max: {arr.max():.4f}")

        return np.expand_dims(arr, axis=0)
    except Exception as e:
        # Si la imagen está corrupta, lanzamos error para capturarlo después
        raise ValueError(f"Error procesando imagen: {e}")

# ENDPOINT: PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    # 1. Seguridad: Verificar que el modelo existe
    if model is None:
        return jsonify({"error": "El modelo no está disponible"}), 500

    # 2. Flexibilidad: Aceptar 'image' O 'file'
    if 'image' not in request.files and 'file' not in request.files:
        return jsonify({"error": "Debes subir una imagen (key 'image' o 'file')"}), 400
    
    file = request.files.get("image") or request.files.get("file")

    # --- Imprimir qué llega ---
    print(f"📸 API Recibió archivo: {file.filename}")
    
    try:
        # 3. Procesamiento
        img_bytes = file.read()
        img_processed = preprocess_image(img_bytes)
        
        # 4. Inferencia
        pred = model.predict(img_processed)

        # --- Predicción Cruda ---
        print(f"📊 [DEBUG PRED] Vector crudo: {pred}")

        class_index = int(np.argmax(pred))
        class_name = LABELS.get(class_index, "Unknown")
        confidence = float(np.max(pred)) # <--- Extraemos la confianza

        # 5. Guardado Seguro en BD
        new_prediction = Prediction(
            filename=file.filename,
            prediction=class_name,
            confidence=confidence
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Al final de la función, cuando guardas:
        print(f"🧠 Predicción: {class_name} | Confianza: {confidence}")

        return jsonify({
            "id": new_prediction.id,
            "timestamp": new_prediction.timestamp.isoformat(), # Asumiendo que tu modelo lo genera
            "filename": file.filename,
            "prediction_name": class_name,
            "confidence": confidence
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400 # Error de usuario (imagen mala)
    except Exception as e:
        db.session.rollback() # <--- ¡IMPORTANTE! Limpiamos la transacción fallida
        return jsonify({"error": f"Error interno: {str(e)}"}), 500

# ENDPOINT: TODAS LAS PREDICCIONES

@app.route("/predictions", methods=["GET"])
def get_predictions():
    preds = Prediction.query.order_by(Prediction.id).all()
    return jsonify([
        {
            "id": p.id,
            "timestamp": p.timestamp.isoformat(),
            "filename": p.filename,
            "prediction": p.prediction,
            "confidence": p.confidence
        }
        for p in preds
    ])

# ENDPOINT: PREDICCIÓN POR ID

@app.route("/predictions/<int:pred_id>", methods=["GET"])
def get_prediction_by_id(pred_id):
    p = Prediction.query.get(pred_id)
    if not p:
        return jsonify({"error": "ID no encontrado"}), 404

    return jsonify({
        "id": p.id,
        "timestamp": p.timestamp.isoformat(),
        "filename": p.filename,
        "prediction": p.prediction,
        "confidence": p.confidence
    })


# ENDPOINT: BORRAR TABLA

@app.route("/predictions/delete", methods=["DELETE"])
def delete_all():
    db.session.query(Prediction).delete()
    db.session.commit()
    return jsonify({"message": "Todas las predicciones eliminadas"}), 200

# ENDPOINT: RESETEAR BASE DE DATOS

@app.route("/reset_db", methods=["POST"])
def reset_db():
    try:
        db.drop_all()
        db.create_all()
        return jsonify({"status": "ok", "message": "Database reset done"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)