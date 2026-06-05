FROM python:3.10-slim

WORKDIR /app

# 1. Evitar bloqueos interactivos durante la instalación
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Reemplazo de libgl1-mesa-glx por libgl1 (Soporte para Debian moderno)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copiar e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar código de la app y el modelo ONNX descargado del bucket
COPY ./app /app/app
COPY yolo26n.onnx /app/yolo26n.onnx

# 5. Configurar puertos e inicio para Cloud Run
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]