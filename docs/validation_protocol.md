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

Los splits deberán ser por paciente para evitar fuga de información entre entrenamiento, validación y prueba. Una imagen de un mismo `Patient ID` no debe aparecer en más de un split, porque eso contaminaría la evaluación con información del mismo paciente.

La división base de Fase 2 será 70% entrenamiento, 15% validación y 15% prueba, calculada sobre pacientes únicos. Después de generar `train.csv`, `val.csv` y `test.csv`, se debe verificar que la intersección de `Patient ID` entre splits sea cero.

Los thresholds por clase se optimizarán solo con datos de validación y se reportarán de forma transparente.

La categoría "No Finding" no se tratará como patología ni como etiqueta de salida del modelo; representa ausencia de hallazgos reportados.

Si una fila de NIH ChestX-ray14 tiene `Finding Labels = "No Finding"`, las 14 etiquetas patológicas del modelo deben quedar en `0`. Esta decisión representa ausencia de hallazgos en la metadata y no agrega una clase adicional.

La validación del proyecto es académica y preliminar. No constituye validación clínica, no debe usarse como evidencia diagnóstica y no reemplaza evaluación médica profesional.
