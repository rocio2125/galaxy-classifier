# src/frontend/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Copiamos el requirements desde la raíz (necesitamos subir un nivel en el contexto luego)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiamos todo el código
COPY . .

# Streamlit usa el puerto 8501 por defecto
EXPOSE 8501

# Comando para ejecutar la app y hacerla visible fuera del contenedor
# --server.address=0.0.0.0 es CRUCIAL para que Docker pueda mostrar la web
# Ajusta la ruta si tu archivo se llama distinto de ui.py
CMD ["streamlit", "run", "src/frontend/ui.py", "--server.port=8501", "--server.address=0.0.0.0"]