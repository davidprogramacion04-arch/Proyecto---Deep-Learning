# Proyecto - Deep Learning

Repositorio del proyecto de predicción de PM2.5 en Bogotá con enfoques de imputación, grafos de estaciones y modelos temporales tipo STN.

## Objetivo

Explorar, comparar y documentar estrategias para pronosticar material particulado PM2.5, evaluando:

- persistencia y baselines simples,
- imputación con Missing Forest,
- uso de información espacial mediante grafos de estaciones,
- modelos LSTM y STN.

## Estructura principal

- `modelos y bd/`: notebooks, checkpoints, dataset y artefactos de análisis.
- `articulos/`: literatura y referencias del proyecto.
- `metadatos/`: documentos complementarios y material de referencia.
- `documentacion/`: resumen ejecutivo y resultados del proyecto.

## Resultados clave

Se documentaron avances en varias etapas:

- Fase 0: persistencia como baseline.
- Fase 1: Random Forest + MissForest.
- Fase 2: Random Forest + MissForest + grafo de estaciones.
- Fase B: LSTM + MissForest + grafo.
- STN replicado para Bogotá y comparado con persistencia.

Los resultados reportados incluyen métricas como RMSE, MAE y R² y se dejan resumidos en:

- `documentacion/resumen_resultados.md`

## Notebooks relevantes

- `modelos y bd/STN_PM25_Fontibon_replica.ipynb`
- `modelos y bd/Baseline_PM25_Fontibon_2Fases_LSTM.ipynb`
- `modelos y bd/Baseline_PM25_Fontibon_3Fases_v3_imputacion_rapida (1).ipynb`
- `modelos y bd/03_MissingForest_Grafos_STN.ipynb`

## Reproducibilidad

Se recomienda usar la configuración de entorno del archivo `deep_learning_pm25_linux.yml` o instalar los paquetes del archivo `requirements.txt`.

## Flujo de trabajo

```bash
git pull
git add .
git commit -m "Añadir resultados y notebook de Missing Forest + grafos"
git push
```

## Nota

Este repositorio incluye el avance del trabajo realizado hasta este punto, con datos, notebooks, checkpoints, resultados y documentación para mantener el progreso en GitHub.
