from src.database.db import db

class Prediction(db.Model):
    __tablename__ = 'predictions'

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    prediction = db.Column(db.String(500), nullable=False)

    def __repr__(self):
        return f'<Prediction {self.id}: {self.filename}>'