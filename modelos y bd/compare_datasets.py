"""
Comparación cuantitativa entre TU dataset (Bogotá/FONTIBÓN) y el dataset público
de Beijing que usa el paper del STN.

Responde con números, no con intuición, a la pregunta "¿me sirve entrenar con
los datos de Kaggle?": lo que decide eso es si las dos series viven en el mismo
régimen (misma escala, misma dinámica, mismas variables), no el tamaño.

Uso
---
python compare_datasets.py --bogota data/bogota_calidad_aire_raw.parquet \
                           --beijing data/PRSA_data_2010.1.1-2014.12.31.csv
(cualquiera de los dos es opcional; si falta uno, se reportan los valores
 publicados del paper como referencia)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from stn_data import BOGOTA_FEATURES_DEFAULT, load_beijing, load_bogota

REFERENCIA_PAPER = {
    "Beijing (paper)": {
        "periodo": "2010-01-01 a 2014-12-31",
        "n_horas": 43800,
        "variables": 8,
        "split": "train 2010-2013 / test 2014",
        "RMSE_STN_h1": 19.04, "MAE_STN_h1": 11.13, "R2_STN_h1": 0.937,
    },
    "Taizhou (paper)": {
        "periodo": "2017-01-01 a 2019-12-31",
        "n_horas": 26000,
        "variables": 8,
        "split": "train 2017-2018 / test 2019",
        "RMSE_STN_h1": 5.79, "MAE_STN_h1": 3.76, "R2_STN_h1": 0.924,
    },
}


def perfil(serie: pd.Series, nombre: str) -> dict:
    s = serie.dropna()
    lag1 = s.autocorr(lag=1)
    lag24 = s.autocorr(lag=24)
    return {
        "dataset": nombre,
        "n_horas_totales": len(serie),
        "cobertura_%": round(100 * len(s) / len(serie), 1),
        "media": round(float(s.mean()), 2),
        "mediana": round(float(s.median()), 2),
        "desv_est": round(float(s.std()), 2),
        "p95": round(float(s.quantile(0.95)), 2),
        "max": round(float(s.max()), 2),
        "coef_variacion": round(float(s.std() / s.mean()), 3),
        "autocorr_lag1": round(float(lag1), 4),
        "autocorr_lag24": round(float(lag24), 4),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bogota", default=None, help="ruta al parquet de Bogotá")
    p.add_argument("--estacion", default="FONTIBÓN")
    p.add_argument("--beijing", default=None, help="ruta al CSV de Kaggle")
    args = p.parse_args()

    filas = []

    if args.bogota:
        df_b = load_bogota(args.bogota, estacion=args.estacion,
                           variables=BOGOTA_FEATURES_DEFAULT)
        filas.append(perfil(df_b["PM2.5"], f"Bogotá — {args.estacion}"))
        print(f"\nCobertura por variable en {args.estacion}:")
        print((df_b.notna().mean() * 100).round(1).to_string())

    if args.beijing:
        df_bj = load_beijing(args.beijing)
        filas.append(perfil(df_bj["pm2.5"], "Beijing (Kaggle/UCI)"))

    if filas:
        tabla = pd.DataFrame(filas).set_index("dataset")
        print("\n" + "=" * 78)
        print("PERFIL COMPARADO DE LA SERIE DE PM2.5")
        print("=" * 78)
        print(tabla.T.to_string())

    print("\n" + "=" * 78)
    print("REFERENCIA PUBLICADA (Zhang & Zhang 2023, Tabla 1, h1)")
    print("=" * 78)
    print(pd.DataFrame(REFERENCIA_PAPER).to_string())

    print("""
Cómo leer esto
--------------
1. autocorr_lag1: si en tu serie es tan alta como en Beijing, la persistencia ya
   explica casi todo y un R2 alto del STN NO es evidencia de que aprendió algo.
   Compará siempre contra la fila de persistencia que imprime train_stn.py.
2. media y desv_est: el R2 y el RMSE no son comparables entre ciudades con
   escalas distintas. Si tu media es ~5x menor que la de Beijing, un RMSE de
   19 ug/m3 allá y uno de 6 aca pueden ser el MISMO desempeño relativo.
3. coef_variacion: es la forma honesta de comparar dificultad entre datasets.
4. cobertura_%: es tu restriccion real, no el numero de horas. Un dataset de
   43.800 horas con 60% de cobertura tiene menos senal utilizable que uno de
   26.000 horas completo.
""")


if __name__ == "__main__":
    main()
