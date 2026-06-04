# Instrucciones de uso

## 1. Ejecutar la aplicación

```bash
cd vehicle-monitoring-onnx
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

En Windows PowerShell:

```powershell
cd vehicle-monitoring-onnx
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 2. Abrir la interfaz

```text
http://127.0.0.1:8000
```

## 3. Descargar/verificar el modelo

```text
http://127.0.0.1:8000/admin/model
```

Presiona **Descargar o actualizar modelo**.

## 4. Realizar una predicción

1. Entra a `http://127.0.0.1:8000`.
2. Carga una imagen de tráfico, vía o parqueadero.
3. Presiona **Contar vehículos**.
4. El sistema muestra el conteo y permite descargar el resultado en JSON o TXT.

## 5. Configurar qué clases contar

Solo carros:

```bash
VEHICLE_CLASS_IDS=2
```

Vehículos en general:

```bash
VEHICLE_CLASS_IDS=2,3,5,7
```

Equivalencias COCO:

```text
2 = car
3 = motorcycle
5 = bus
7 = truck
```

## 6. Probar API técnica

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@trafico.jpg"
```

## 7. Historial

```text
http://127.0.0.1:8000/download-history
```

La aplicación guarda localmente las predicciones en `logs/` si no se configura un bucket.
