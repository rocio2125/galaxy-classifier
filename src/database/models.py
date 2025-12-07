from src.database.db import db
from datetime import datetime

class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(500), nullable=False)
    confidence = db.Column(db.Float, nullable=True)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.filename}>'