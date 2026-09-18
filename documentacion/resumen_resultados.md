# Proyecto Deep Learning - PM2.5 en Bogotá

## Resumen ejecutivo

Este repositorio consolidó la comparación de estrategias de imputación y modelado para predicción de PM2.5 en la estación Fontibón, siguiendo una lógica incremental:

- Fase 0: persistencia / baseline simple
- Fase 1: Random Forest + MissForest (intra-estación)
- Fase 2: Random Forest + MissForest + grafo de vecindad entre estaciones
- Fase 3: LSTM con imputación + grafo
- STN sobre la serie temporal con comparación frente a persistencia

La intención fue evaluar si la información espacial y la imputación multivariada aportan valor frente a un modelo simple, y si el STN ofrece una mejora real sobre un baseline riguroso.

## Resultados principales reportados

### Baseline / modelos previos

| Fase | Descripción | RMSE | MAE | R² | Observación |
|---|---|---:|---:|---:|---|
| Fase 0 | Persistencia | ~7.5 | ~5.0 | ~0.65 | Línea base del problema |
| Fase 1 | RF + MissForest | ~7.1 | ~4.8 | ~0.70 | Mejora respecto a persistencia |
| Fase 2 | RF + MissForest + grafo | ~7.0 | ~4.7 | ~0.688 | Sube un poco respecto a Fase 1 |
| Fase B | LSTM + MissForest + grafo | 6.398 | 4.439 | 0.7401 | Mejor rendimiento entre los baselines |

### STN

| Fuente | RMSE | MAE | R² | Comentario |
|---|---:|---:|---:|---|
| Beijing (paper) - STN | 19.04 | 11.13 | 0.937 | Referencia del paper |
| Taizhou (paper) - STN | 5.79 | 3.76 | 0.924 | Referencia fuerte |
| Bogotá FONTIBÓN - STN | 5.991 | 4.209 | 0.7722 | Resultado del STN sobre la estación de estudio |
| Bogotá FONTIBÓN - Persistencia | ~6.34 | ~4.39 | ~0.74 | Baseline de comparación |

### Nueva etapa: STN con imputación vecinal

- Se preparó una base dedicada de FONTIBÓN en formato horario para el STN usando la imputación por vecindad y el grafo de estaciones.
- La base nueva se guarda como `fontibon_vecinal_stn_ready.parquet` y se almacena fuera de GitHub para evitar peso excesivo en el repo.
- La configuración revisada del STN quedó definida con la mejor configuración encontrada en la búsqueda previa y se deja lista para ejecución en un cuaderno independiente.
- Esta base y su entrenamiento quedan fuera del repositorio principal y se gestionan desde la carpeta de Drive del proyecto.

## Interpretación

- El STN en Bogotá logra un rendimiento claramente competitivo frente a la persistencia y al menos comparable a varias estrategias de Machine Learning.
- La diferencia entre STN y persistencia es razonable, pero no debe confundirse con una validación absoluta de calidad; hay que comparar siempre sobre el mismo split, la misma métrica y el mismo horizonte de predicción.
- El modelo con grafo y MissForest mejora la representación espacial de los datos y ayuda a mitigar pérdida de información en estaciones con huecos temporales.
- El mejor resultado global del proyecto se observa en la combinación de LSTM + MissForest + grafo, aunque el STN sigue siendo una línea muy relevante para investigación posterior.

## Archivos clave del repo

- `modelos y bd/STN_PM25_Fontibon_replica.ipynb`: notebook central de ejecución del STN y análisis reproducible.
- `modelos y bd/Baseline_PM25_Fontibon_2Fases_LSTM.ipynb`: comparación de baselines y etapas de imputación.
- `modelos y bd/Baseline_PM25_Fontibon_3Fases_v3_imputacion_rapida (1).ipynb`: exploración complementaria y gráficas de fase.
- `modelos y bd/bogota_calidad_aire_raw.parquet`: datos de entrada principales.
- `modelos y bd/ckpt_beijing.pt` y `modelos y bd/ckpt_bogota_stn.pt`: checkpoints entrenados.

## Siguientes pasos recomendados

1. Reforzar la comparación cuantitativa con un esquema de validación por ventana temporal.
2. Evaluar Missing Forest con más variables, no solo PM2.5.
3. Construir un grafo de estaciones en formato estructura de red para análisis nodal.
4. Hacer una comparación STN vs LSTM vs Random Forest en el mismo escenario de datos.
5. Exportar resultados a tabla PDF o CSV para incluirlos en la documentación final.

## Estado del proyecto

El proyecto ya quedó en una etapa avanzada, con notebooks ejecutados, artefactos generados y resultados documentados. Este resumen sirve como base para dejar el trabajo listo para un commit y almacenamiento en GitHub.
