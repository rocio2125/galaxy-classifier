from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
import numpy as np
from keras.models import load_model
from PIL import Image
import io
import os


# CONFIG FLASK

app = Flask(__name__)

app.config["SQLALCHEMY_DATABASE_URI"] = (
    "postgresql://rizzijp:6bxbCP9beZdlqj2gQRMsly0NNOhAmPcw@"
    "dpg-d4q53cc9c44c73b6o2jg-a.frankfurt-postgres.render.com/"
    "predictionsdb_4sex?sslmode=require"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


# MODELO SQLALCHEMY

class Prediction(db.Model):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    filename = Column(String, nullable=False)
    prediction = Column(String, nullable=False)

with app.app_context():
    db.create_all()


# CARGA DEL MODELO

MODEL_PATH = "../models/modelo_galaxias.keras"
model = load_model(MODEL_PATH)

LABELS = {
    0: "spiral",
    1: "elliptical",
    2: "lenticular",
    3: "irregular",
    4: "merger",
    5: "unknown",
    6: "barred spiral",
    7: "compact",
    8: "edge-on",
    9: "uncertain"
}


# PREPROCESADO

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)



# ENDPOINT: PREDICT

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Debes subir una imagen"}), 400

    image_file = request.files["image"]
    filename = image_file.filename
    img_bytes = image_file.read()

    img = preprocess_image(img_bytes)
    pred = model.predict(img)
    class_index = int(np.argmax(pred))
    class_name = LABELS[class_index]

    # Guardar en BD
    new_prediction = Prediction(
        filename=filename,
        prediction=class_name
    )
    db.session.add(new_prediction)
    db.session.commit()

    return jsonify({
        "id": new_prediction.id,
        "timestamp": new_prediction.timestamp.isoformat(),
        "filename": filename,
        "prediction_name": class_name
    })


# ENDPOINT: TODAS LAS PREDICCIONES

@app.route("/predictions", methods=["GET"])
def get_predictions():
    preds = Prediction.query.order_by(Prediction.id).all()
    return jsonify([
        {
            "id": p.id,
            "timestamp": p.timestamp.isoformat(),
            "filename": p.filename,
            "prediction": p.prediction
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
        "prediction": p.prediction
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

# MAIN

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")