# 1. Usar una imagen base oficial de Python ligera
FROM python:3.10-slim

# 2. Configurar el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Copiar las librerías necesarias e instalarlas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el código fuente de tu API (FastAPI)
COPY ./app /app/app

# 5. CUMPLIMIENTO DEL REQUERIMIENTO:
# Copiamos el modelo .onnx que el pipeline de GitHub descargó 
# dinámicamente desde el bucket hacia adentro del contenedor.
COPY yolo26n.onnx /app/yolo26n.onnx

# 6. Exponer el puerto que usa FastAPI (Cloud Run usa el 8080 o el 8000)
EXPOSE 8000

# 7. Comando para ejecutar la aplicación cuando el contenedor se encienda
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
