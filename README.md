# Tesis_CXR

Sistema web desplegado para la clasificación preliminar multi-etiqueta de hallazgos radiológicos torácicos en radiografías de tórax mediante arquitectura híbrida CNN-ViT y explicabilidad visual.

**Disclaimer obligatorio:** Resultado preliminar de uso académico. No constituye diagnóstico clínico.

## Alcance

Este repositorio corresponde a un proyecto académico de Ingeniería de Software. No es un sistema clínico real, no diagnostica pacientes, no reemplaza al radiólogo y no es software médico certificado.

La tarea final del proyecto será la clasificación preliminar multi-etiqueta sobre radiografías de tórax usando las 14 etiquetas patológicas de NIH ChestX-ray14. La categoría "No Finding" no se trata como patología; solo representa ausencia de hallazgos reportados y no se incluye en `artifacts/labels.json`.

En Fase 1 existen placeholders para entrenamiento, inferencia y explicabilidad visual. En Fase 2 se agregan scripts y notebook base para preparar el dataset NIH ChestX-ray14 localmente o en Kaggle. No se descarga NIH desde el repositorio, no se entrena un modelo y no se ejecuta inferencia real.

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

## Fase 2 — Preparación de dataset en Kaggle

El dataset NIH ChestX-ray14 no se sube al repositorio. Las imágenes deben agregarse como **Input** en Kaggle y los scripts deben ejecutarse apuntando a las rutas del entorno de Kaggle. Los scripts solo generan archivos CSV de metadata; no copian imágenes ni entrenan modelos.

Archivos generados esperados:

- `nih_multilabel.csv`
- `train.csv`
- `val.csv`
- `test.csv`

Ejemplo para preparar metadata multilabel en Kaggle:

```bash
python scripts/prepare_nih_multilabel.py \
  --metadata-csv /kaggle/input/datasets/organizations/nih-chest-xrays/data/Data_Entry_2017.csv \
  --images-root /kaggle/input/datasets/organizations/nih-chest-xrays/data \
  --output-csv /kaggle/working/processed/nih_multilabel.csv \
  --recursive-image-search
```

Ejemplo para crear splits por paciente:

```bash
python scripts/create_patient_splits.py \
  --input-csv /kaggle/working/processed/nih_multilabel.csv \
  --output-dir /kaggle/working/processed/splits
```

`No Finding` no es una etiqueta del modelo. Si una fila tiene `Finding Labels = "No Finding"`, las 14 etiquetas patológicas quedan en `0`.

Las variantes de escritura del CSV de NIH, como `Pleural_Thickening`, se normalizan a la forma canonica usada por `artifacts/labels.json`: `Pleural Thickening`.

## Fase 3 — Validacion de Dataset PyTorch

Antes de entrenar se valida que los CSV generados puedan alimentar `NIHMultilabelDataset` y un `DataLoader` de PyTorch. Esta fase solo prueba lectura de imagenes, transformaciones basicas, vector multi-label y formas de batch; no entrena modelos.

Comando Kaggle:

```bash
python scripts/check_dataset.py \
  --csv-path /kaggle/working/processed/splits/train.csv \
  --labels-json artifacts/labels.json \
  --batch-size 8 \
  --num-workers 2
```

Salida esperada:

- `Batch images shape: [8, 3, 224, 224]`
- `Batch targets shape: [8, 14]`
- confirmacion de que `No Finding` no aparece como etiqueta ni columna.

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
