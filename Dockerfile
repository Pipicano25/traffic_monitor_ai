FROM python:3.10-slim

WORKDIR /app

# 1. Configurar variables de entorno indispensables para evitar bloqueos interactivos de APT
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# 2. Comando apt-get robusto y corregido para evitar el Exit Code 100
RUN apt-get clean && apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Continuar con el resto de tu configuración normal...
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /app/app
COPY yolo26n.onnx /app/yolo26n.onnx

ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]