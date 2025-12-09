import os
import io
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv
import tensorflow as tf # Usamos TF solo para el intérprete Lite
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

# 3. CARGA DEL MODELO LITE (Optimizado para memoria baja)
# CÁLCULO DE LA RUTA DEL MODELO
# Obtenemos la ruta de ESTE archivo (app.py): .../src/api/app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Subimos un nivel para llegar a 'src': .../src
SRC_DIR = os.path.dirname(BASE_DIR)
# Entramos a 'models' y apuntamos al archivo
MODEL_PATH = os.path.join(SRC_DIR, 'models', 'modelo_galaxias.tflite')

interpreter = None
input_details = None
output_details = None

print("⏳ Cargando modelo TFLite...")
try:
    # Cargamos el intérprete (ocupa 10 veces menos RAM)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    # Obtenemos referencias de entrada y salida
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print("🚀 Modelo TFLite cargado correctamente!")
except Exception as e:
    print(f"❌ Error cargando TFLite: {e}")

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

# PREPROCESADO
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        #arr = np.array(img) / 255.0
        # PROCESADO TFLITE: Float32 y rango 0-255
        arr = np.array(img, dtype=np.float32)

        # LOG DE DIAGNÓSTICO
        print(f"🔍 [DEBUG IMG] Shape: {arr.shape} | Min: {arr.min():.4f} | Max: {arr.max():.4f}")
        # Dimensión extra para batch: (1, 224, 224, 3)
        return np.expand_dims(arr, axis=0)
    except Exception as e:
        # Si la imagen está corrupta, lanzamos error para capturarlo después
        raise ValueError(f"Error procesando imagen: {e}")

# 5. ENDPOINTS

# ENDPOINT: BIENVENIDA
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API Lite Activa"}), 200

# ENDPOINT: ESTADO DE LA API
@app.route("/health", methods=["GET"])
def health():
    if interpreter is None:
         return jsonify({"status": "error", "reason": "Model not loaded"}), 500
    return jsonify({"status": "ok", "type": "TFLite"}), 200

# ENDPOINT: PREDICT
@app.route("/predict", methods=["POST"])
def predict():
    if interpreter is None:
        return jsonify({"error": "Modelo no disponible"}), 500

    if 'image' not in request.files and 'file' not in request.files:
        return jsonify({"error": "Falta imagen"}), 400
    
    file = request.files.get("image") or request.files.get("file")
    print(f"📸 Recibido: {file.filename}")

    try:
        img_bytes = file.read()
        input_data = preprocess_image(img_bytes)

        # --- INFERENCIA TFLITE ---
        # 1. Poner los datos en la entrada
        interpreter.set_tensor(input_details[0]['index'], input_data)
        # 2. Ejecutar
        interpreter.invoke()
        # 3. Leer la salida
        pred = interpreter.get_tensor(output_details[0]['index'])
        # -------------------------

        print(f"📊 Vector: {pred}")
        
        class_index = int(np.argmax(pred))
        class_name = LABELS.get(class_index, "Unknown")
        confidence = float(np.max(pred))

        # Guardar
        new_prediction = Prediction(
            filename=file.filename,
            prediction=class_name,
            confidence=confidence
        )
        db.session.add(new_prediction)
        db.session.commit()

        return jsonify({
            "id": new_prediction.id,
            "filename": file.filename,
            "prediction_name": class_name,
            "confidence": confidence
        })

    except Exception as e:
        db.session.rollback()
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500


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

# ENDPOINT: PREDICCIÓN POR FILENAME

@app.route("/predictions/filename", methods=["GET"])
def get_by_filename():
    filename = request.args.get("filename", "").strip()

    if not filename:
        return jsonify({"error": "filename requerido"}), 400

    # Case-insensitive + parcial:
    results = (
        Prediction.query
        .filter(Prediction.filename.ilike(f"%{filename}%"))
        .all()
    )

    if not results:
        return jsonify({"error": "Archivo no encontrado"}), 404

    return jsonify([r.to_dict() for r in results]), 200

# ENDPOINT: BORRAR TABLA

@app.route("/predictions/delete", methods=["DELETE"])
def delete_all():
    db.session.query(Prediction).delete()
    db.session.commit()
    return jsonify({"message": "Todas las predicciones eliminadas"}), 200

# ENDPOINT: RESETEAR BASE DE DATOS

@app.route("/predictions/reset", methods=["DELETE"])
def delete_all_predictions():
    from sqlalchemy import text
    db.session.execute(text("TRUNCATE TABLE predictions RESTART IDENTITY CASCADE;"))
    db.session.commit()
    return jsonify({"status": "ok", "message": "Tabla reseteada y IDs reiniciados"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)