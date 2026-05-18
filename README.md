# Tesis_CXR

Sistema web desplegado para la clasificación preliminar multi-etiqueta de hallazgos radiológicos torácicos en radiografías de tórax mediante arquitectura híbrida CNN-ViT y explicabilidad visual.

**Disclaimer obligatorio:** Resultado preliminar de uso académico. No constituye diagnóstico clínico.

## Alcance

Este repositorio corresponde a un proyecto académico de Ingeniería de Software. No es un sistema clínico real, no diagnostica pacientes, no reemplaza al radiólogo y no es software médico certificado.

La tarea final del proyecto será la clasificación preliminar multi-etiqueta sobre radiografías de tórax usando las 14 etiquetas patológicas de NIH ChestX-ray14. La categoría "No Finding" no se trata como patología; solo representa ausencia de hallazgos reportados y no se incluye en `artifacts/labels.json`.

En esta Fase 1 solo existen placeholders para entrenamiento, inferencia y explicabilidad visual. No se descarga NIH, no se entrena un modelo y no se ejecuta inferencia real.

## Arquitectura Resumida

Usuario -> UI Streamlit -> API FastAPI -> modelo PyTorch futuro -> thresholds -> Grad-CAM futuro -> JSON -> UI Streamlit.

Componentes iniciales:

- `api/`: servicio FastAPI con `/health`, `/model-info` y `/predict` placeholder.
- `ui/`: interfaz Streamlit mínima con login básico, carga de archivos y verificación de salud de la API.
- `src/`: módulos base para datasets, modelos, entrenamiento, evaluación, explicabilidad y utilidades.
- `artifacts/`: etiquetas NIH, thresholds y configuraciones JSON iniciales.
- `docs/`: documentación inicial de arquitectura, validación y despliegue.

## Ejecutar Localmente

Crear entorno virtual:

```powershell
py -3.12 -m venv .venv
```

Activar entorno:

```powershell
.\.venv\Scripts\Activate.ps1
```

Instalar dependencias mínimas:

```powershell
pip install -r api/requirements.txt
pip install -r ui/requirements.txt
```

Ejecutar API:

```powershell
cd api
uvicorn main:app --reload --port 8000
```

Ejecutar UI en otra terminal:

```powershell
cd ui
streamlit run app.py --server.port 8501
```

La UI usará por defecto:

- usuario: `admin`
- contraseña: `admin123`
- API: `http://127.0.0.1:8000`

## Ejecutar Con Docker

Desde la raíz del repositorio:

```powershell
docker compose up --build
```

Servicios:

- API: `http://127.0.0.1:8080`
- UI: `http://127.0.0.1:8501`

## Endpoints Iniciales

```text
GET /health
GET /model-info
POST /predict
```

`POST /predict` devuelve HTTP 501 porque la inferencia real se implementará en fases posteriores.

## Datos y Privacidad

No se guardan pacientes, imágenes, predicciones históricas ni reportes. Para fases futuras, solo se aceptará DICOM anonimizado y los archivos subidos no deberán persistirse.

## Entrenamiento Futuro

`requirements-train.txt`, `configs/` y `scripts/` quedan preparados como base para fases posteriores. El modelo real `.pt` no se incluye en esta fase.

