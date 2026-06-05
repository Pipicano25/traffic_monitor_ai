FROM python:3.10-slim

WORKDIR /app

# 1. Evitar bloqueos interactivos y configurar salida limpia
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Instalar dependencias nativas del sistema operativo para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Copiar e instalar las dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. SOLUCIÓN COMPLETA: Copiamos TODO el proyecto (incluyendo static, templates, app y scripts)
COPY . .

# 5. Colocamos el modelo .onnx descargado del bucket en la raíz del contenedor
COPY yolo26n.onnx /app/yolo26n.onnx

# 6. Variables de entorno indispensables para Cloud Run
ENV PORT=8080
EXPOSE 8080

# 7. EJECUCIÓN: Agregamos la raíz al PYTHONPATH para que encuentre el módulo 'app' 
# y todas las carpetas adyacentes de inmediato.
CMD ["sh", "-c", "PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]