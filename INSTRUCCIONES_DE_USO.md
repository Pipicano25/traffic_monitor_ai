# Instrucciones de uso

Este documento describe cómo ejecutar, probar y consumir la aplicación **Traffic Monitor AI**, una solución de inferencia basada en **FastAPI + ONNX Runtime** para detectar y contar vehículos en imágenes. También explica cómo validar el modelo localmente, cómo usar la API y cómo verificar el historial de predicciones por ambiente.

---

## 1. Requisitos previos

Antes de ejecutar el proyecto, asegúrate de contar con:

- Python 3.10 o superior.
- Git.
- Docker, si deseas ejecutar la aplicación en contenedor.
- Acceso a Google Cloud CLI, si deseas descargar artefactos desde Cloud Storage o validar servicios desplegados en Cloud Run.
- Credenciales o permisos sobre el proyecto de Google Cloud, si vas a operar contra GCP.

Para validar la instalación de Python:

```bash
python --version
```

Para validar Docker:

```bash
docker --version
```

Para validar Google Cloud CLI:

```bash
gcloud --version
```

---

## 2. Clonar el repositorio

```bash
git clone https://github.com/Pipicano25/traffic_monitor_ai.git
cd traffic_monitor_ai
```

Selecciona la rama que deseas usar:

```bash
git checkout dev
```

o para producción:

```bash
git checkout prod
```

> La rama `dev` está asociada al ambiente de desarrollo y la rama `prod` al ambiente productivo.

---

## 3. Crear el entorno virtual

### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Variables de entorno principales

La aplicación utiliza variables de entorno para definir el modelo, el ambiente, los umbrales de inferencia, las clases a contar y el bucket donde se registran las predicciones.

| Variable | Descripción | Valor sugerido |
|---|---|---|
| `MODEL_URL` | URL externa desde donde se puede descargar el modelo ONNX. | `https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx` |
| `MODEL_PATH` | Ruta local donde estará disponible el modelo ONNX. | `models/yolo26n.onnx` |
| `ENV_STAGE` | Ambiente de ejecución usado para separar logs. | `dev` o `prod` |
| `GCP_BUCKET_NAME` | Bucket donde se guardan las predicciones. | `traffic-mlops-storage` |
| `CONF_THRESHOLD` | Umbral mínimo de confianza para aceptar detecciones. | `0.25` |
| `IOU_THRESHOLD` | Umbral IoU usado en postprocesamiento/NMS cuando aplica. | `0.45` |
| `VEHICLE_CLASS_IDS` | Clases COCO que serán contadas como vehículos. | `2` o `2,3,5,7` |
| `TEST_IMAGE_PATH`| Imagen de prueba|"tests/data/img1.jpeg"|
| `EXPECTED_COUNT`| Numero de carros esperado|"20"|
| `MAX_ABS_ERROR`| Margen de error |"2"|

Ejemplo para Linux / macOS:

```bash
export MODEL_PATH="models/yolo26n.onnx"
export ENV_STAGE="dev"
export GCP_BUCKET_NAME="traffic-mlops-storage"
export CONF_THRESHOLD="0.25"
export IOU_THRESHOLD="0.45"
export VEHICLE_CLASS_IDS="2"
export TEST_IMAGE_PATH: "tests/data/img1.jpeg"
export EXPECTED_COUNT: "20"
export MAX_ABS_ERROR: "2"
```

Ejemplo para Windows PowerShell:

```powershell
$env:MODEL_PATH="models/yolo26n.onnx"
$env:ENV_STAGE="dev"
$env:GCP_BUCKET_NAME="traffic-mlops-storage"
$env:CONF_THRESHOLD="0.25"
$env:IOU_THRESHOLD="0.45"
$env:VEHICLE_CLASS_IDS="2"
```

---

## 5. Clases vehiculares utilizadas

El modelo trabaja con clases del dataset COCO. Para el caso del monitoreo vehicular, se recomienda usar:

```text
2 = car
3 = motorcycle
5 = bus
7 = truck
```

Si solo deseas contar carros:

```bash
VEHICLE_CLASS_IDS=2
```

Si deseas contar vehículos en general:

```bash
VEHICLE_CLASS_IDS=2,3,5,7
```

---

## 6. Descargar o verificar el modelo ONNX

El archivo `.onnx` **no debe almacenarse directamente en el repositorio**. Puede obtenerse desde la interfaz de administración o desde Cloud Storage.

### Opción A: desde la interfaz web

Ejecuta la aplicación y abre:

```text
http://127.0.0.1:8000/admin/model
```

Luego presiona:

```text
Descargar o actualizar modelo
```

### Opción B: desde Google Cloud Storage

Si tienes permisos sobre el bucket:

```bash
mkdir -p models
gcloud storage cp gs://traffic-mlops-storage/models/yolo26n.onnx ./models/yolo26n.onnx
```

Verifica que el archivo exista:

```bash
ls -lh models/yolo26n.onnx
```

---

## 7. Ejecutar la aplicación localmente

Con el entorno virtual activo, ejecuta:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Abre la interfaz web:

```text
http://127.0.0.1:8000
```

También puedes consultar la documentación automática de FastAPI:

```text
http://127.0.0.1:8000/docs
```

---

## 8. Realizar una predicción desde la interfaz web

1. Ingresa a `http://127.0.0.1:8000`.
2. Carga una imagen de tráfico, calle, vía o parqueadero.
3. Presiona **Contar vehículos**.
4. El sistema retorna:
   - Conteo total de vehículos.
   - Conteo por clase.
   - Coordenadas de las detecciones.
   - Imagen anotada.
   - Archivo JSON descargable.
   - Archivo TXT descargable.

---

## 9. Consumir la API con `curl`

Para consumir el endpoint técnico `/predict`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -F "file=@trafico.jpg"
```

Ejemplo de respuesta esperada:

```json
{
  "request_id": "uuid-generado",
  "timestamp_utc": "2026-06-08T00:00:00+00:00",
  "environment": "DEV",
  "model_path": "models/yolo26n.onnx",
  "vehicle_class_ids": [2],
  "vehicle_class_labels": ["car"],
  "confidence_threshold": 0.25,
  "count": 20,
  "class_counts": {
    "car": 20
  },
  "detections": []
}
```

> El número de detecciones puede variar según la imagen, el umbral de confianza y las clases configuradas.

---

## 10. Endpoints principales

| Método | Endpoint | Uso |
|---|---|---|
| `GET` | `/` | Interfaz web para cargar imágenes y visualizar resultados. |
| `POST` | `/ui/predict` | Endpoint usado internamente por la interfaz web. |
| `POST` | `/predict` | Endpoint API para realizar inferencia desde otros sistemas. |
| `GET` | `/docs` | Documentación Swagger generada automáticamente por FastAPI. |
| `GET` | `/health` | Validación básica del estado del servicio. |
| `GET` | `/model-status` | Estado del modelo ONNX configurado. |
| `GET` | `/admin/model` | Interfaz administrativa para verificar o descargar el modelo. |
| `POST` | `/admin/download-model` | Descarga o actualiza el modelo ONNX desde la fuente configurada. |
| `GET` | `/download-history` | Descarga el historial TXT de predicciones del ambiente actual. |

---

## 11. Consultar el estado del servicio

Para verificar que la aplicación responde correctamente:

```bash
curl http://127.0.0.1:8000/health
```

Respuesta esperada:

```json
{
  "status": "ok",
  "environment": "DEV",
  "stage_mlops": "DEV",
  "model_path": "models/yolo26n.onnx",
  "interface": "enabled",
  "annotation": "enabled"
}
```

Para verificar el estado del modelo:

```bash
curl http://127.0.0.1:8000/model-status
```

---

## 12. Historial de predicciones

Cada llamada al endpoint `/predict` o a la interfaz web registra una línea en formato JSON dentro de un archivo TXT.

En ambiente de desarrollo:

```text
logs/predicciones_dev.txt
```

En ambiente de producción:

```text
logs/predicciones_prod.txt
```

Desde la aplicación puedes descargar el historial en:

```text
http://127.0.0.1:8000/download-history
```

Si la aplicación se ejecuta con acceso a Google Cloud, los logs se escriben en Cloud Storage. Si no hay credenciales o bucket disponible, el sistema guarda una copia local en la carpeta `logs/`.

---

## 13. Ejecutar pruebas localmente

Las pruebas validan que el modelo responda correctamente y que el conteo no tenga un cambio significativo frente a una métrica esperada.

### Preparar datos de prueba

```bash
mkdir -p tests/data
gcloud storage cp gs://traffic-mlops-storage/test-data/img1.jpeg ./tests/data/img1.jpeg
gcloud storage cp gs://traffic-mlops-storage/test-data/img2.jpeg ./tests/data/img2.jpeg
```

### Preparar el modelo

```bash
mkdir -p models
gcloud storage cp gs://traffic-mlops-storage/models/yolo26n.onnx ./models/yolo26n.onnx
```

### Ejecutar pruebas

```bash
export MODEL_PATH="models/yolo26n.onnx"
export TEST_IMAGE_PATH="tests/data/img1.jpeg"
export EXPECTED_COUNT="20"
export MAX_ABS_ERROR="2"

pytest -v
```

En Windows PowerShell:

```powershell
$env:MODEL_PATH="models/yolo26n.onnx"
$env:TEST_IMAGE_PATH="tests/data/img1.jpeg"
$env:EXPECTED_COUNT="20"
$env:MAX_ABS_ERROR="2"

pytest -v
```

Las pruebas principales verifican:

- Que el modelo responda con una entrada definida.
- Que la respuesta tenga la estructura esperada.
- Que el conteo esté dentro de un umbral aceptable frente al valor esperado.
- Que las detecciones incluyan coordenadas válidas.

---

## 14. Ejecutar con Docker

Para construir la imagen Docker, el archivo `yolo26n.onnx` debe estar en la raíz del proyecto, porque el `Dockerfile` lo copia dentro del contenedor en `/app/yolo26n.onnx`.

Descarga el modelo para construcción:

```bash
gcloud storage cp gs://traffic-mlops-storage/models/yolo26n.onnx ./yolo26n.onnx
```

Construye la imagen:

```bash
docker build -t traffic-monitor-app:local .
```

Ejecuta el contenedor:

```bash
docker run --rm -p 8080:8080 \
  -e MODEL_PATH="/app/yolo26n.onnx" \
  -e ENV_STAGE="dev" \
  -e GCP_BUCKET_NAME="traffic-mlops-storage" \
  -e CONF_THRESHOLD="0.25" \
  -e IOU_THRESHOLD="0.45" \
  -e VEHICLE_CLASS_IDS="2" \
  traffic-monitor-app:local
```

Abre la aplicación:

```text
http://127.0.0.1:8080
```

Prueba la API:

```bash
curl -X POST "http://127.0.0.1:8080/predict" \
  -H "accept: application/json" \
  -F "file=@trafico.jpg"
```

---

## 15. Uso de endpoints desplegados en Cloud Run

El pipeline de GitHub Actions despliega servicios separados según la rama:

```text
traffic-monitor-service-dev
traffic-monitor-service-prod
```

Para obtener las URLs desde Google Cloud:

```bash
gcloud run services describe traffic-monitor-service-dev \
  --region us-central1 \
  --format="value(status.url)"
```

```bash
gcloud run services describe traffic-monitor-service-prod \
  --region us-central1 \
  --format="value(status.url)"
```

Luego puedes consumir los endpoints así:

```bash
curl -X POST "https://URL_DEL_SERVICIO_DEV/predict" \
  -H "accept: application/json" \
  -F "file=@trafico.jpg"
```

```bash
curl -X POST "https://URL_DEL_SERVICIO_PROD/predict" \
  -H "accept: application/json" \
  -F "file=@trafico.jpg"
```

---

## 16. Despliegue automático con GitHub Actions

El pipeline se ejecuta automáticamente cuando se realiza un `push` a las ramas:

```text
dev
prod
```

El flujo general es:

1. Se descarga el código del repositorio.
2. Se configura Python.
3. Se instalan dependencias.
4. Se autentica contra Google Cloud mediante Workload Identity Federation.
5. Se descargan imágenes de prueba desde Cloud Storage.
6. Se descarga el modelo ONNX desde Cloud Storage.
7. Se ejecutan pruebas con `pytest`.
8. Si las pruebas pasan, se construye la imagen Docker.
9. Se publica la imagen en Artifact Registry.
10. Se actualiza el servicio correspondiente en Cloud Run.

Ambientes:

| Rama | Servicio Cloud Run | Archivo de logs |
|---|---|---|
| `dev` | `traffic-monitor-service-dev` | `logs/predicciones_dev.txt` |
| `prod` | `traffic-monitor-service-prod` | `logs/predicciones_prod.txt` |

---

## 17. Buenas prácticas de uso

- No subir el archivo `.onnx` al repositorio.
- No subir imágenes de prueba pesadas al repositorio.
- No subir archivos generados en `outputs/`.
- No subir logs locales generados en `logs/`.
- Usar variables de entorno para separar `dev` y `prod`.
- Validar localmente con `pytest -v` antes de hacer `push`.
- Usar `dev` para pruebas y `prod` solo para versiones estables.

---

## 18. Solución de problemas frecuentes

### Error: no se encontró el modelo ONNX

Verifica la variable `MODEL_PATH` y descarga el modelo:

```bash
mkdir -p models
gcloud storage cp gs://traffic-mlops-storage/models/yolo26n.onnx ./models/yolo26n.onnx
export MODEL_PATH="models/yolo26n.onnx"
```

También puedes descargarlo desde:

```text
http://127.0.0.1:8000/admin/model
```

### Error: el puerto ya está en uso

Cambia el puerto de ejecución:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8001
```

Luego abre:

```text
http://127.0.0.1:8001
```

### Error: no se pueden subir logs a Cloud Storage

Valida que exista el bucket y que las credenciales estén disponibles:

```bash
gcloud auth login
gcloud config set project traffic-monitor-mlops
gcloud storage ls gs://traffic-mlops-storage
```

Si no hay acceso a GCP, la aplicación seguirá guardando una copia local en `logs/`.

### Error: Docker no encuentra `yolo26n.onnx`

Antes de construir la imagen, descarga el modelo en la raíz del proyecto:

```bash
gcloud storage cp gs://traffic-mlops-storage/models/yolo26n.onnx ./yolo26n.onnx
docker build -t traffic-monitor-app:local .
```

---

## 19. Flujo recomendado para desarrollo

```bash
git checkout dev
git pull origin dev

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p models tests/data
gcloud storage cp gs://traffic-mlops-storage/models/yolo26n.onnx ./models/yolo26n.onnx
gcloud storage cp gs://traffic-mlops-storage/test-data/img1.jpeg ./tests/data/img1.jpeg

export MODEL_PATH="models/yolo26n.onnx"
export TEST_IMAGE_PATH="tests/data/img1.jpeg"
export EXPECTED_COUNT="20"
export MAX_ABS_ERROR="2"

pytest -v
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Si todo funciona correctamente:

```bash
git add .
git commit -m "feat: actualizar aplicación de monitoreo vehicular"
git push origin dev
```

El `push` a `dev` activará el pipeline automático de pruebas, construcción y despliegue del servicio de desarrollo.

---

## 20. Flujo recomendado para producción

Cuando la rama `dev` esté validada:

```bash
git checkout prod
git pull origin prod
git merge dev
git push origin prod
```

El `push` a `prod` activará el pipeline y actualizará el servicio productivo en Cloud Run.
