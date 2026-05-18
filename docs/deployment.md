# Despliegue

**Disclaimer obligatorio:** Resultado preliminar de uso académico. No constituye diagnóstico clínico.

## Ejecución Local

API FastAPI:

```powershell
cd api
uvicorn main:app --reload --port 8000
```

UI Streamlit:

```powershell
cd ui
streamlit run app.py --server.port 8501
```

## Docker Compose

Desde la raíz del repositorio:

```powershell
docker compose up --build
```

Servicios:

- API: `http://127.0.0.1:8080`
- UI: `http://127.0.0.1:8501`

## Despliegue Futuro

En fases posteriores se podrá evaluar despliegue en Google Cloud Run usando contenedores separados para API y UI. La imagen de API incorporará inferencia real solo cuando exista un modelo entrenado y validado académicamente.

## Privacidad y Persistencia

El sistema no debe persistir imágenes, datos sensibles, pacientes, predicciones históricas ni reportes. Para DICOM, solo se aceptarán archivos anonimizados y se procesarán sin almacenamiento permanente.

