from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from urllib.parse import quote


def render_home(environment: str, conf_threshold: float, vehicle_class_ids: list[int], model_status: dict | None = None) -> str:
    model_ready = bool(model_status and model_status.get("ready"))
    model_text = "Listo" if model_ready else "Pendiente"
    model_size = int(model_status.get("size_bytes", 0)) if model_status else 0
    model_size_mb = round(model_size / (1024 * 1024), 2)
    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Monitoreo vehicular ONNX</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card: #111827;
      --soft: #1f2937;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --primary: #38bdf8;
      --primary-dark: #0284c7;
      --ok: #22c55e;
      --border: #334155;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: Arial, Helvetica, sans-serif;
      background: radial-gradient(circle at top, #1e3a8a 0, var(--bg) 45%);
      color: var(--text);
    }}
    .container {{ width: min(980px, 92%); margin: 0 auto; padding: 46px 0; }}
    .hero {{ display: grid; grid-template-columns: 1.1fr .9fr; gap: 24px; align-items: stretch; }}
    .card {{
      background: rgba(17, 24, 39, 0.92);
      border: 1px solid var(--border);
      border-radius: 24px;
      padding: 28px;
      box-shadow: 0 20px 60px rgba(0,0,0,.35);
    }}
    h1 {{ font-size: clamp(30px, 5vw, 52px); margin: 0 0 12px; line-height: 1.04; }}
    h2 {{ margin-top: 0; font-size: 24px; }}
    p {{ color: var(--muted); line-height: 1.6; }}
    .badge {{ display: inline-flex; gap: 8px; align-items: center; background: rgba(56,189,248,.12); color: #bae6fd; border: 1px solid rgba(56,189,248,.32); padding: 8px 12px; border-radius: 999px; margin-bottom: 16px; }}
    .upload-box {{ border: 2px dashed var(--border); background: rgba(15, 23, 42, .65); border-radius: 20px; padding: 22px; }}
    input[type=file] {{ width: 100%; padding: 14px; border: 1px solid var(--border); border-radius: 14px; background: var(--soft); color: var(--text); }}
    button, .button {{
      display: inline-block;
      margin-top: 16px;
      border: 0;
      border-radius: 14px;
      background: linear-gradient(135deg, var(--primary), var(--primary-dark));
      color: white;
      padding: 13px 18px;
      font-weight: 700;
      cursor: pointer;
      text-decoration: none;
    }}
    .button.secondary {{ background: var(--soft); border: 1px solid var(--border); }}
    .flow {{ margin-top: 24px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }}
    .step {{ background: rgba(31, 41, 55, .75); border: 1px solid var(--border); border-radius: 16px; padding: 16px; }}
    .step strong {{ color: white; display: block; margin-bottom: 8px; }}
    .meta {{ display: grid; gap: 10px; margin-top: 18px; }}
    .meta div {{ display:flex; justify-content:space-between; gap:16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; color: var(--muted); }}
    .meta span {{ color: var(--text); font-weight: 700; }}
    footer {{ margin-top: 24px; color: var(--muted); font-size: 14px; }}
    @media (max-width: 820px) {{ .hero, .flow {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <main class="container">
    <section class="hero">
      <div class="card">
        <div class="badge">🚗 Sistema inteligente de monitoreo vehicular</div>
        <h1>Conteo automático vehicular con modelo ONNX</h1>
        <p>
          Esta interfaz permite que un usuario final suba una imagen de una vía o parqueadero,
          el sistema ejecute el modelo YOLO en formato ONNX y devuelva únicamente la cantidad de vehículos detectados,
          sin dibujar cajas ni señalar los vehículos.
        </p>
        <div class="flow">
          <div class="step"><strong>1. Cargar</strong>El usuario selecciona una imagen.</div>
          <div class="step"><strong>2. Predecir</strong>La app envía la imagen al modelo.</div>
          <div class="step"><strong>3. Contar</strong>El sistema cuenta vehículos detectados.</div>
          <div class="step"><strong>4. Registrar</strong>Guarda la predicción en TXT.</div>
        </div>
      </div>

      <div class="card">
        <h2>Realizar predicción</h2>
        <form class="upload-box" action="/ui/predict" method="post" enctype="multipart/form-data">
          <label for="file">Imagen de entrada</label>
          <br><br>
          <input id="file" name="file" type="file" accept="image/*" required>
          <button type="submit">Contar vehículos</button>
        </form>
        <a class="button secondary" href="/download-history">Descargar historial TXT</a>
        <a class="button secondary" href="/admin/model">Verificar modelo ONNX</a>
        <a class="button secondary" href="/docs">Ver API técnica</a>
        <div class="meta">
          <div>Ambiente actual <span>{html.escape(environment)}</span></div>
          <div>Clases contadas <span>{html.escape(str(vehicle_class_ids))}</span></div>
          <div>Confianza mínima <span>{conf_threshold}</span></div>
          <div>Modelo ONNX <span>{model_text}</span></div>
          <div>Tamaño modelo <span>{model_size_mb} MB</span></div>
        </div>
      </div>
    </section>
    <footer>
      El mismo contenedor funciona para dev y prod. La rama define el endpoint y el archivo TXT asociado.
    </footer>
  </main>
</body>
</html>
"""


def render_result(response: dict, image_data_uri: str | None = None) -> str:
    result_json = json.dumps(response, indent=2, ensure_ascii=False)
    result_txt = json.dumps({"timestamp_utc": datetime.now(timezone.utc).isoformat(), **response}, ensure_ascii=False)

    json_href = "data:application/json;charset=utf-8," + quote(result_json)
    txt_href = "data:text/plain;charset=utf-8," + quote(result_txt + "\n")

    count = int(response.get("count", 0))
    environment = html.escape(str(response.get("environment", "")))
    request_id = html.escape(str(response.get("request_id", "")))
    confidence = html.escape(str(response.get("confidence_threshold", "")))
    class_counts = response.get("class_counts", {}) or {}
    class_counts_text = html.escape(json.dumps(class_counts, ensure_ascii=False))
    vehicle_labels = response.get("vehicle_class_labels", []) or []
    vehicle_labels_text = html.escape(", ".join(str(x) for x in vehicle_labels))

    preview = ""
    if image_data_uri:
        preview = f'<img class="preview" src="{image_data_uri}" alt="Imagen cargada sin anotaciones">'

    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Resultado - Monitoreo vehicular</title>
  <style>
    :root {{ --bg:#0f172a; --card:#111827; --soft:#1f2937; --text:#e5e7eb; --muted:#9ca3af; --primary:#38bdf8; --primary-dark:#0284c7; --border:#334155; --ok:#22c55e; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; min-height:100vh; font-family:Arial, Helvetica, sans-serif; background:radial-gradient(circle at top, #164e63 0, var(--bg) 45%); color:var(--text); }}
    .container {{ width:min(1050px,92%); margin:0 auto; padding:42px 0; }}
    .grid {{ display:grid; grid-template-columns:.9fr 1.1fr; gap:24px; }}
    .card {{ background:rgba(17,24,39,.94); border:1px solid var(--border); border-radius:24px; padding:28px; box-shadow:0 20px 60px rgba(0,0,0,.35); }}
    .count {{ font-size:92px; font-weight:900; line-height:1; color:var(--ok); margin:12px 0; }}
    h1, h2 {{ margin-top:0; }}
    p {{ color:var(--muted); line-height:1.55; }}
    .preview {{ width:100%; border-radius:18px; border:1px solid var(--border); margin-top:16px; }}
    .button {{ display:inline-block; margin:8px 8px 0 0; border-radius:14px; background:linear-gradient(135deg,var(--primary),var(--primary-dark)); color:white; padding:12px 16px; font-weight:700; text-decoration:none; }}
    .button.secondary {{ background:var(--soft); border:1px solid var(--border); }}
    pre {{ background:#020617; border:1px solid var(--border); border-radius:18px; padding:18px; overflow:auto; color:#d1fae5; }}
    .meta {{ display:grid; gap:10px; margin:18px 0; }}
    .meta div {{ display:flex; justify-content:space-between; gap:16px; border-bottom:1px solid var(--border); padding-bottom:8px; color:var(--muted); }}
    .meta span {{ color:var(--text); font-weight:700; }}
    @media (max-width:850px) {{ .grid {{ grid-template-columns:1fr; }} .count {{ font-size:72px; }} }}
  </style>
</head>
<body>
  <main class="container">
    <div class="grid">
      <section class="card">
        <h1>Resultado del monitoreo</h1>
        <p>El sistema analizó la imagen y devolvió el conteo sin señalar los vehículos visualmente.</p>
        <div class="count">{count}</div>
        <p>Vehículos detectados</p>
        <div class="meta">
          <div>Ambiente <span>{environment}</span></div>
          <div>Confianza mínima <span>{confidence}</span></div>
          <div>Clases contadas <span>{vehicle_labels_text}</span></div>
          <div>Conteo por clase <span>{class_counts_text}</span></div>
          <div>ID de petición <span>{request_id[:18]}...</span></div>
        </div>
        <a class="button" href="{json_href}" download="resultado_prediccion.json">Descargar resultado JSON</a>
        <a class="button" href="{txt_href}" download="resultado_prediccion.txt">Descargar resultado TXT</a>
        <a class="button secondary" href="/download-history">Descargar historial TXT</a>
        <a class="button secondary" href="/">Nueva predicción</a>
        {preview}
      </section>
      <section class="card">
        <h2>Respuesta técnica</h2>
        <p>Este JSON es el mismo que consumiría otro sistema si se conecta directamente al endpoint <strong>/predict</strong>.</p>
        <pre>{html.escape(result_json)}</pre>
      </section>
    </div>
  </main>
</body>
</html>
"""


def render_error(message: str) -> str:
    safe = html.escape(message)
    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Error - Monitoreo vehicular</title>
  <style>
    body {{ margin:0; min-height:100vh; display:grid; place-items:center; font-family:Arial, Helvetica, sans-serif; background:#0f172a; color:#e5e7eb; }}
    .card {{ width:min(720px,92%); background:#111827; border:1px solid #334155; border-radius:24px; padding:28px; }}
    p {{ color:#fca5a5; line-height:1.6; }}
    a {{ color:white; display:inline-block; margin-top:16px; background:#0284c7; padding:12px 16px; border-radius:12px; text-decoration:none; }}
  </style>
</head>
<body>
  <section class="card">
    <h1>No fue posible procesar la imagen</h1>
    <p>{safe}</p>
    <a href="/">Volver a intentar</a>
  </section>
</body>
</html>
"""


def render_model_admin(status: dict, message: str | None = None) -> str:
    ready = bool(status.get("ready"))
    exists = bool(status.get("exists"))
    size_bytes = int(status.get("size_bytes", 0))
    size_mb = round(size_bytes / (1024 * 1024), 2)
    model_path = html.escape(str(status.get("model_path", "")))
    model_url = html.escape(str(status.get("model_url", "")))
    state = "Listo para predecir" if ready else "Pendiente por descargar"
    message_html = ""
    if message:
        message_html = f'<div class="message">{html.escape(message)}</div>'

    return f"""
<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Administrar modelo ONNX</title>
  <style>
    :root {{ --bg:#0f172a; --card:#111827; --soft:#1f2937; --text:#e5e7eb; --muted:#9ca3af; --primary:#38bdf8; --primary-dark:#0284c7; --border:#334155; --ok:#22c55e; --warn:#f59e0b; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; font-family:Arial, Helvetica, sans-serif; background:radial-gradient(circle at top,#1e3a8a 0,var(--bg) 45%); color:var(--text); }}
    .container {{ width:min(900px,92%); margin:0 auto; padding:46px 0; }}
    .card {{ background:rgba(17,24,39,.94); border:1px solid var(--border); border-radius:24px; padding:28px; box-shadow:0 20px 60px rgba(0,0,0,.35); }}
    h1 {{ margin-top:0; font-size:clamp(28px,4vw,44px); }}
    p {{ color:var(--muted); line-height:1.6; }}
    .status {{ display:inline-block; padding:8px 12px; border-radius:999px; background:{'#064e3b' if ready else '#78350f'}; color:white; font-weight:700; }}
    .meta {{ display:grid; gap:12px; margin:22px 0; }}
    .meta div {{ display:grid; grid-template-columns:180px 1fr; gap:14px; border-bottom:1px solid var(--border); padding-bottom:10px; color:var(--muted); }}
    .meta span {{ color:var(--text); font-weight:700; overflow-wrap:anywhere; }}
    button, .button {{ display:inline-block; border:0; border-radius:14px; background:linear-gradient(135deg,var(--primary),var(--primary-dark)); color:white; padding:13px 18px; font-weight:700; cursor:pointer; text-decoration:none; margin-right:8px; margin-top:8px; }}
    .button.secondary {{ background:var(--soft); border:1px solid var(--border); }}
    .message {{ margin:16px 0; padding:14px 16px; background:rgba(56,189,248,.12); border:1px solid rgba(56,189,248,.35); border-radius:14px; color:#bae6fd; }}
    .note {{ margin-top:18px; padding:16px; background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.35); border-radius:16px; color:#fde68a; }}
    @media (max-width:700px) {{ .meta div {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="container">
    <section class="card">
      <h1>Administrar modelo ONNX</h1>
      <p>Esta pantalla permite verificar si el modelo existe localmente y descargarlo desde la referencia externa definida en <strong>MODEL_URL</strong>.</p>
      <div class="status">{state}</div>
      {message_html}
      <div class="meta">
        <div>Existe archivo <span>{'Sí' if exists else 'No'}</span></div>
        <div>Preparado <span>{'Sí' if ready else 'No'}</span></div>
        <div>Tamaño <span>{size_mb} MB</span></div>
        <div>Ruta local <span>{model_path}</span></div>
        <div>URL del modelo <span>{model_url}</span></div>
      </div>
      <form action="/admin/download-model" method="post">
        <button type="submit">Descargar o actualizar modelo</button>
        <a class="button secondary" href="/">Volver a la interfaz</a>
        <a class="button secondary" href="/model-status">Ver estado JSON</a>
      </form>
      <div class="note">
        Recomendación: esta ruta es útil para desarrollo o demostración académica. En producción debe estar protegida con autenticación/IAM, o el modelo debe descargarse automáticamente en CI/CD durante la construcción del contenedor.
      </div>
    </section>
  </main>
</body>
</html>
"""
