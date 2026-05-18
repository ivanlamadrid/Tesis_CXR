# Protocolo de Validación

**Disclaimer obligatorio:** Resultado preliminar de uso académico. No constituye diagnóstico clínico.

## Objetivo

La validación futura será académica y preliminar. No constituye validación clínica, no certifica seguridad médica y no habilita uso diagnóstico.

## Métricas Futuras

Se evaluará la tarea multi-etiqueta con métricas apropiadas para las 14 etiquetas patológicas de NIH ChestX-ray14:

- AUC por clase.
- Macro AUC.
- Micro AUC.
- F1.
- Precisión.
- Sensibilidad.
- Especificidad.

## Consideraciones

Los splits deberán ser por paciente para evitar fuga de información entre entrenamiento, validación y prueba. Los thresholds por clase se optimizarán solo con datos de validación y se reportarán de forma transparente.

La categoría "No Finding" no se tratará como patología ni como etiqueta de salida del modelo; representa ausencia de hallazgos reportados.

