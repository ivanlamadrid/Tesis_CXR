# Arquitectura

**Disclaimer obligatorio:** Resultado preliminar de uso académico. No constituye diagnóstico clínico.

## Flujo Principal

```text
Usuario
  -> Streamlit
  -> FastAPI
  -> modelo PyTorch futuro
  -> thresholds
  -> Grad-CAM futuro
  -> JSON
  -> Streamlit
```

## Componentes

- `ui/`: interfaz web mínima con Streamlit. Permite login básico, carga de archivo y visualización de placeholders.
- `api/`: servicio FastAPI. Expone salud del servicio, metadatos del modelo y un endpoint `/predict` aún no implementado.
- `artifacts/`: configuraciones iniciales, etiquetas NIH y thresholds placeholder.
- `src/`: código base para futuras fases de entrenamiento, inferencia, evaluación y explicabilidad.
- `data/`: carpetas reservadas para datos locales. No se versionan imágenes médicas ni datos procesados.

## Alcance de Fase 1

La Fase 1 no ejecuta entrenamiento, inferencia ni Grad-CAM real. El modelo PyTorch se integrará en fases posteriores. No existe base de datos y no se persisten imágenes, pacientes, predicciones ni reportes.

