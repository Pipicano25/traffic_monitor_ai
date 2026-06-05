# 1. Usar una imagen oficial de Python ligera
FROM python:3.10-slim

# 2. Instalar dependencias del sistema operativo indispensables para OpenCV / Dibujo de imágenes
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Establecer el directorio de trabajo raíz
WORKDIR /app

# 4. Copiar e instalar requerimientos primero (Aprovecha la caché de Docker)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar los archivos respetando de manera estricta la estructura de FastAPI
# Copiamos la lógica de la app
COPY ./app /app/app

# Copiamos las carpetas de recursos visuales exigidas en tu main.py a la raíz /app
COPY ./static /app/static
COPY ./templates /app/templates

# 6. Requerimiento del profesor: Insertar el modelo descargado del bucket en la raíz
COPY yolo26n.onnx /app/yolo26n.onnx

# 7. Crear directorios necesarios para las salidas locales temporales
RUN mkdir -p /app/outputs

# 8. Variables de entorno por defecto para producción
ENV PORT=8080
ENV ENVIRONMENT="production"
ENV MODEL_PATH="/app/yolo26n.onnx"

EXPOSE 8080

# 9. Comando de ejecución optimizado: Lee de forma obligatoria el $PORT asignado por Cloud Run
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]