# STN para PM2.5 — réplica ejecutable

> **Empezá por `STN_PM25_Fontibon_replica.ipynb`.** Ese cuaderno es autocontenido: trae la
> arquitectura, los datos, la validación en Beijing, la búsqueda de hiperparámetros y el modelo
> final, todo en secciones numeradas. Los `.py` de esta carpeta quedan como alternativa para
> correr desde la terminal (útil si vas a lanzar una búsqueda larga en segundo plano); el
> contenido es el mismo.

Implementación desde cero del modelo **STN** (*sparse attention-based Transformer
networks*) de Zhang & Zhang (2023), *Int. J. Environ. Sci. Technol.* 20:13535–13550,
DOI [10.1007/s13762-023-04900-1](https://doi.org/10.1007/s13762-023-04900-1) —
el paper cuyos R² de 0.937 (Beijing) y 0.924 (Taizhou) ya usás como referencia en
el cuaderno baseline.

No existe código ni pesos oficiales de los autores. Esta réplica sigue el paper
donde el paper es explícito y sigue **Informer** (Zhou et al., AAAI 2021), que el
propio paper cita como base de la atención dispersa, donde el paper no lo es.

---

## 1. Qué está en el paper y qué tuvimos que inducir

| Componente | ¿En el paper? | Valor |
|---|---|---|
| Encoder-decoder con position embedding senoidal | Sí | — |
| Atención multi-head dispersa (K-L + top-u queries, O(L·log L)) | Sí, con las ecuaciones | — |
| Destilación Conv1D k=3 + max-pool stride 2 | Sí | — |
| Encoder de **dos ramas** (una recibe ½·Lx) | Sí | — |
| Decoder: atención dispersa enmascarada + atención estándar cruzada + FFN + lineal | Sí | — |
| Nº de capas | Sí | 3 encoders, 2 decoders |
| Lookback | Sí | 24 h |
| Optimización | Sí | Adam, lr 1e-4, batch 32, ≤200 épocas, MSE, datos en [0,1] |
| Split | Sí | año-independiente (últimos 12 meses = test) |
| **d_model, n_heads, d_ff, dropout, factor c, label_len** | **No** | defaults de Informer: 512, 8, 2048, 0.05, 5 |
| Cómo se codifica la dirección del viento | No | one-hot (nuestra decisión) |
| Normalización por variable vs. global | No | MinMax por variable, ajustado solo en train |

Todo lo de la fila "No" es exactamente lo que conviene meter en la búsqueda de
hiperparámetros: no hay valor de referencia que replicar.

---

## 2. Instalación (VS Code local)

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Con `seq_len=24` y ~44.000 horas, el modelo corre en **CPU** sin problema
(minutos por época con d_model=512; segundos con d_model=128). GPU solo hace
falta si vas a hacer una búsqueda amplia.

Estructura sugerida:

```
stn_pm25/
├── data/
│   ├── PRSA_data_2010.1.1-2014.12.31.csv     # Beijing (Kaggle)
│   └── bogota_calidad_aire_raw.parquet       # tu caché del EDA
├── stn_model.py      # arquitectura
├── stn_data.py       # carga, interpolación, split, ventaneo
├── train_stn.py      # entrenar + evaluar (incluye baseline de persistencia)
├── search_stn.py     # búsqueda de hiperparámetros con origen rodante
└── compare_datasets.py
```

Dataset de Beijing: `kaggle datasets download -d djhavera/beijing-pm25-data-data-set`
(es el mismo archivo de UCI "Beijing PM2.5 Data"; el paper enlaza esa misma
página de Kaggle). El de Taizhou **no** es público.

---

## 3. Orden de trabajo recomendado

**Paso 1 — validar que la implementación es correcta (en Beijing).**

```bash
python train_stn.py --dataset beijing \
  --data-path data/PRSA_data_2010.1.1-2014.12.31.csv \
  --test-start 2014-01-01 --pred-len 1 --epochs 30
```

Si el R² sobre 2014 queda alrededor de 0.93, la réplica es fiel y podés citarla.
Si queda muy por debajo, el problema es la implementación, no tus datos — y
conviene descubrirlo acá y no sobre Bogotá.

**Paso 2 — comparar los dos datasets antes de decidir nada.**

```bash
python compare_datasets.py --bogota data/bogota_calidad_aire_raw.parquet \
                           --beijing data/PRSA_data_2010.1.1-2014.12.31.csv
```

**Paso 3 — buscar hiperparámetros, solo dentro del train.**

```bash
python search_stn.py --dataset bogota --data-path data/bogota_calidad_aire_raw.parquet \
  --test-start 2024-01-01 --n-trials 20 --epochs 12 --folds 3 \
  --out-json busqueda_stn_bogota.json
```

**Paso 4 — entrenar el modelo final con la mejor config y evaluar 2024 una sola vez.**

```bash
python train_stn.py --dataset bogota --data-path data/bogota_calidad_aire_raw.parquet \
  --estacion FONTIBÓN --test-start 2024-01-01 --pred-len 1 \
  --d-model 256 --n-heads 8 --d-ff 512 --dropout 0.1 --lr 3e-4 --epochs 60 \
  --out-json resultado_stn_bogota.json
```

**Paso 5 (opcional) — transfer learning Beijing → Bogotá.**

```bash
python train_stn.py --dataset beijing ... --save-ckpt ckpt_beijing.pt
python train_stn.py --dataset bogota  ... --init-ckpt ckpt_beijing.pt --lr 1e-5
```

Solo se transfieren las capas cuyo tamaño coincide; las de entrada/salida se
reinicializan porque el número y el significado de las variables es distinto.
Trátalo como un experimento con hipótesis nula ("no ayuda"), no como el plan A.

---

## 4. Colab

```python
!pip -q install torch pandas pyarrow
from google.colab import files; files.upload()   # subí los .py y el parquet
!python train_stn.py --dataset bogota --data-path bogota_calidad_aire_raw.parquet \
    --test-start 2024-01-01 --pred-len 1 --epochs 30
```

Con GPU (T4), una época sobre ~35.000 ventanas tarda unos pocos segundos.

---

## 5. Reglas que el código respeta (las mismas de tu cuaderno baseline)

1. Split temporal, nunca aleatorio.
2. Escalador MinMax ajustado **solo** con train.
3. La validación por defecto es el tramo final del train (`--val-mode tail`),
   no un 10% aleatorio. El 10% aleatorio del paper está disponible como
   `--val-mode random` para poder mostrar la diferencia, pero infla la métrica.
4. En test se evalúa **solo donde el PM2.5 real está observado**: las horas
   interpoladas entran como insumo del modelo pero nunca como verdad de terreno.
5. `train_stn.py` siempre imprime el baseline de persistencia junto al STN.
   Si el STN no le saca una diferencia clara, no hay nada que festejar.

## 6. Advertencia honesta

El código está escrito y verificado sintácticamente, pero **no pudo ejecutarse
de punta a punta en este entorno** (sin PyTorch ni acceso a tus datos). La
primera corrida real es el Paso 1: sirve justamente como prueba de que la
implementación funciona.
