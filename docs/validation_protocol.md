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

Antes de iniciar entrenamiento se debe validar el `Dataset` y el `DataLoader` de PyTorch. La verificacion minima incluye:

- Lectura correcta de imagenes desde `image_path`.
- Conversion de imagen a RGB y tensor de dimensiones `3x224x224`.
- Vector multi-label `float32` de 14 posiciones.
- Ausencia de `No Finding` como etiqueta del modelo o columna de salida.
- Batch consistente con forma `[batch, 3, 224, 224]` para imagenes y `[batch, 14]` para targets.

El baseline DenseNet121 se usa como modelo de referencia antes de desarrollar la arquitectura CNN-ViT. Sus metricas permiten establecer una linea base reproducible para comparar mejoras posteriores. La evaluacion del baseline debe reportar perdida de validacion, AUC por clase cuando sea posible, macro AUC, micro AUC, F1 macro, precision macro y recall macro.

El checkpoint del baseline se selecciona por mayor `macro_auc` cuando esta metrica existe; si no puede calcularse, se selecciona por menor `val_loss`. `No Finding` no debe aparecer como etiqueta durante entrenamiento ni evaluacion.

La validación del proyecto es académica y preliminar. No constituye validación clínica, no debe usarse como evidencia diagnóstica y no reemplaza evaluación médica profesional.
