# /Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Instalar las dependencias basicas
RUN apt-get update && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Exponer puerto interno
EXPOSE 10000

# Arrancar Flask con Gunicorn
CMD ["gunicorn", "src.api.app:app", "--bind", "0.0.0.0:10000"]