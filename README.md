# Sistema inteligente de monitoreo vehicular con ONNX

Aplicación FastAPI con interfaz web para contar vehículos en imágenes usando un modelo YOLO26 en formato ONNX.

El modelo **no se guarda en el repositorio**. Se descarga desde una referencia externa definida en `MODEL_URL`:

```bash
https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx
```

## Corrección importante de inferencia

Esta versión soporta dos formatos de salida ONNX:

1. YOLO26 end-to-end / NMS-free: `[1, 300, 6]`, con `x1, y1, x2, y2, score, class_id`.
2. YOLO clásico: `[1, 84, 8400]` o `[1, 8400, 84]`, con cajas y scores por clase.

La versión anterior podía devolver `0` porque interpretaba la salida `[300, 6]` como si fuera una matriz clásica de scores, cuando realmente YOLO26 puede entregar detecciones finales ya procesadas.

## Clases COCO usadas

Por defecto se cuenta la clase:

```bash
VEHICLE_CLASS_IDS=2
```

Donde `2 = car`.

Si deseas monitoreo vehicular amplio, usa:

```bash
VEHICLE_CLASS_IDS=2,3,5,7
```

Donde:

```text
2 = car
3 = motorcycle
5 = bus
7 = truck
```

## Lanzar localmente

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

Luego abre:

```text
http://127.0.0.1:8000
```

## Descargar/verificar modelo desde interfaz

```text
http://127.0.0.1:8000/admin/model
```

Desde esa ruta puedes descargar el modelo ONNX sin ejecutar comandos manuales.

## Endpoints principales

| Ruta | Uso |
|---|---|
| `/` | Interfaz web para cargar imagen y contar vehículos |
| `/ui/predict` | Endpoint usado por la interfaz |
| `/predict` | Endpoint API para otros sistemas |
| `/docs` | Swagger / documentación técnica |
| `/admin/model` | Verificar o descargar el modelo |
| `/model-status` | Estado JSON del modelo |
| `/download-history` | Descargar historial TXT |

## Probar con Docker

```bash
docker build \
  --build-arg MODEL_URL="https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx" \
  -t vehicle-monitoring-onnx .
```

```bash
docker run -p 8080:8080 \
  -e ENVIRONMENT=dev \
  -e MODEL_PATH=models/yolo26n.onnx \
  -e CONF_THRESHOLD=0.25 \
  -e IOU_THRESHOLD=0.45 \
  -e VEHICLE_CLASS_IDS=2 \
  -e PREDICTIONS_BLOB=predicciones_dev.txt \
  vehicle-monitoring-onnx
```

Abre:

```text
http://127.0.0.1:8080
```

## GitHub Actions

El workflow se ejecuta con push a:

```text
dev
prod
```

Incluye dos etapas mínimas:

1. `test`: descarga modelo, descarga datos de prueba desde bucket y ejecuta pruebas.
2. `build_promote`: construye imagen Docker y despliega el servicio correspondiente.

## Variables recomendadas en GitHub Actions

```text
MODEL_URL=https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx
GCP_REGION=us-central1
ARTIFACT_REPOSITORY=mlops-containers
PREDICTIONS_BUCKET=tu-bucket-mlops
TEST_DATA_URI=gs://tu-bucket-mlops/test-data/traffic_test.jpg
EXPECTED_METRICS_URI=gs://tu-bucket-mlops/test-data/expected_metrics.json
CONF_THRESHOLD=0.25
IOU_THRESHOLD=0.45
VEHICLE_CLASS_IDS=2
```

## Historial de predicciones

Cada llamada registra una línea JSON en un archivo TXT.

Ambiente dev:

```text
predicciones_dev.txt
```

Ambiente prod:

```text
predicciones_prod.txt
```
