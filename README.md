# Proyecto - Deep Learning

Repositorio del proyecto de predicción de PM2.5 en Bogotá con enfoques de imputación, grafos de estaciones y modelos temporales tipo STN.

## Objetivo

Explorar, comparar y documentar estrategias para pronosticar material particulado PM2.5, evaluando:

- persistencia y baselines simples,
- imputación con Missing Forest,
- uso de información espacial mediante grafos de estaciones,
- modelos LSTM y STN.

## Estructura principal

- `modelos y bd/`: notebooks y artefactos de análisis.
- `articulos/`: literatura y referencias del proyecto.
- `metadatos/`: documentos complementarios y material de referencia.
- `documentacion/`: resumen ejecutivo y resultados del proyecto.

## Archivos grandes en Drive (no en GitHub)

Los siguientes archivos se almacenan fuera del repositorio para evitar el límite de GitHub y se pueden descargar desde la carpeta de Drive del proyecto:

- `modelos y bd/bogota_calidad_aire_raw.parquet`
- `modelos y bd/fontibon_vecinal_stn_ready.parquet`
- `modelos y bd/ckpt_beijing.pt`
- `modelos y bd/ckpt_bogota_stn.pt`

Enlace de la carpeta compartida del proyecto:

- [Google Drive - Datos y modelos del proyecto](https://drive.google.com/drive/folders/1Ty6nDnn9KBttHslSTWBLYe81qX1F0Nsy?usp=drive_link)

## Qué sí va a GitHub

Los notebooks, PDFs, tablas y documentación del proyecto quedan en el repositorio, incluyendo:

- `modelos y bd/STN_PM25_Fontibon_replica.ipynb`
- `modelos y bd/Baseline_PM25_Fontibon_2Fases_LSTM.ipynb`
- `modelos y bd/Baseline_PM25_Fontibon_3Fases_v3_imputacion_rapida (1).ipynb`
- `modelos y bd/03_MissingForest_Grafos_STN.ipynb`
- `documentacion/resumen_resultados.md`
- `documentacion/resumen_resultados.pdf`
- `documentacion/resumen_resultados.csv`

## Resultados clave

Se documentaron avances en varias etapas:

- Fase 0: persistencia como baseline.
- Fase 1: Random Forest + MissForest.
- Fase 2: Random Forest + MissForest + grafo de estaciones.
- Fase B: LSTM + MissForest + grafo.
- STN replicado para Bogotá y comparado con persistencia.

## Reproducibilidad

Se recomienda usar la configuración de entorno del archivo `deep_learning_pm25_linux.yml` o instalar los paquetes del archivo `requirements.txt`.

## Flujo de trabajo

```bash
git pull
git add .
git commit -m "Actualizar resultados y documentación"
git push
```

## Nota

Este repositorio mantiene la parte esencial del proyecto en GitHub, mientras los datasets y checkpoints pesados quedan en Drive para no romper el límite de almacenamiento del repositorio remoto.
