from flask_sqlalchemy import SQLAlchemy

# Inicializamos el objeto db vacío.
# Se conectará a la app más tarde en el app.py usando init_app()
db = SQLAlchemy()