"""Datos, protocolo común de evaluación y estadística (solo numpy/pandas; sin torch)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

BEIJING_FEATURES = ["pm2.5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir",
                    "cbwd_NE", "cbwd_NW", "cbwd_SE", "cbwd_cv"]
BOGOTA_FEATURES = ["PM2.5", "TAire2", "TAire10", "HAire2", "HAire10",
                   "P", "DViento", "VViento", "PLiquida"]


def load_beijing(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    df["fecha_hora"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"], hour=df["hour"]))
    df = df.set_index("fecha_hora").sort_index()
    dummies = pd.get_dummies(df["cbwd"], prefix="cbwd").astype("float32")
    df = pd.concat([df.drop(columns=["No", "year", "month", "day", "hour", "cbwd"]), dummies], axis=1)
    for col in BEIJING_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    return df[BEIJING_FEATURES].astype("float32")


def load_bogota(parquet_path, estacion="FONTIBÓN", variables=None,
                fecha_inicio="2020-01-01", fecha_fin_exclusiva="2025-01-01") -> pd.DataFrame:
    variables = list(variables or BOGOTA_FEATURES)
    df = pd.read_parquet(parquet_path)
    if {"med_fecha_inicio", "msfl_code", "med_concentracion_estandar"}.issubset(df.columns):
        df["med_fecha_inicio"] = pd.to_datetime(df["med_fecha_inicio"])
        df["med_concentracion_estandar"] = pd.to_numeric(df["med_concentracion_estandar"], errors="coerce")
        sub = df[(df["nombre_est"] == estacion) & (df["msfl_code"].isin(variables))]
        ancho = sub.pivot_table(index="med_fecha_inicio", columns="msfl_code",
                                values="med_concentracion_estandar", aggfunc="mean")
    else:
        ancho = df.copy()
        ancho.index = pd.to_datetime(ancho.index)
    grilla = pd.date_range(fecha_inicio, fecha_fin_exclusiva, freq="h", inclusive="left")
    ancho = ancho.reindex(grilla)
    ancho.index.name = "fecha_hora"
    return ancho[[c for c in variables if c in ancho.columns]].astype("float32")


@dataclass
class Prepared:
    train: np.ndarray            # (n_train, F) escalado
    val_ctx: np.ndarray          # seq_len horas de contexto (cola de train) + val
    test_ctx: np.ndarray         # seq_len horas de contexto (cola de val) + test
    obs_val: np.ndarray          # (n_val,) PM2.5 observado en cada hora de val
    obs_test: np.ndarray         # (n_test,) PM2.5 observado en cada hora de test
    miss_val_ctx: np.ndarray     # (n_ctx+n_val,) PM2.5 faltante (imputado) por fila
    miss_test_ctx: np.ndarray
    scaler_min: np.ndarray
    scaler_max: np.ndarray
    train_std: np.ndarray        # desviación estándar de train en unidades reales
    target_idx: int
    feature_names: list
    test_index: pd.DatetimeIndex
    val_index: pd.DatetimeIndex
    seq_len: int

    def inverse_target(self, y):
        lo, hi = self.scaler_min[self.target_idx], self.scaler_max[self.target_idx]
        return y * (hi - lo) + lo


def prepare_v2(df, target, test_start, seq_len=24, val_fraction=0.1, max_interp_gap=24) -> Prepared:
    """Mismo preprocesamiento causal de la réplica (ffill<=24h + mediana de train, MinMax solo train),
    PERO val y test llevan `seq_len` horas de contexto del tramo anterior: así TODA hora de val/test
    es objetivo de una ventana y el conteo de evaluación es igual para todos los modelos."""
    df = df.copy()
    ti = list(df.columns).index(target)
    is_test = df.index >= pd.Timestamp(test_start)
    observed = df[target].notna().to_numpy()
    tv, te = df.loc[~is_test], df.loc[is_test]
    n_val = max(1, int(len(tv) * val_fraction))
    tr_raw, va_raw = tv.iloc[:-n_val], tv.iloc[-n_val:]
    respaldo = tr_raw.median(numeric_only=True).reindex(df.columns).fillna(0.0)

    def imputar(bloque, contexto=None):
        completo = bloque if contexto is None else pd.concat([contexto, bloque])
        completo = completo.ffill(limit=max_interp_gap).fillna(respaldo)
        return completo.iloc[-len(bloque):], completo.tail(1)

    tr, ctx = imputar(tr_raw); va, ctx = imputar(va_raw, ctx); tt, _ = imputar(te, ctx)
    tr, va, tt = (x.to_numpy("float32") for x in (tr, va, tt))
    lo, hi = tr.min(0), tr.max(0)
    rg = np.where(hi - lo == 0, 1.0, hi - lo)
    sc = lambda a: ((a - lo) / rg).astype("float32")
    tr, va, tt = sc(tr), sc(va), sc(tt)

    obs_tr = observed[: len(tr_raw)]
    obs_va = observed[len(tr_raw): len(tr_raw) + n_val]
    obs_te = observed[is_test]
    return Prepared(
        train=tr,
        val_ctx=np.concatenate([tr[-seq_len:], va]), test_ctx=np.concatenate([va[-seq_len:], tt]),
        obs_val=obs_va, obs_test=obs_te,
        miss_val_ctx=~np.concatenate([obs_tr[-seq_len:], obs_va]),
        miss_test_ctx=~np.concatenate([obs_va[-seq_len:], obs_te]),
        scaler_min=lo, scaler_max=lo + rg,
        train_std=np.nanstd(tr_raw.to_numpy("float64"), axis=0),
        target_idx=ti, feature_names=list(df.columns),
        test_index=df.index[is_test], val_index=va_raw.index, seq_len=seq_len)


def make_windows(arr, seq_len, label_len, pred_len, target_idx):
    """x_enc (N,seq,F), x_dec (N,label+pred,F) [historia + ceros], y (N,pred,1); N = len(arr)-seq_len-pred_len+1."""
    F = arr.shape[1]
    N = len(arr) - seq_len - pred_len + 1
    enc = sliding_window_view(arr, (seq_len, F))[:N, 0]
    hist = sliding_window_view(arr, (label_len, F))[seq_len - label_len: seq_len - label_len + N, 0]
    dec = np.concatenate([hist, np.zeros((N, pred_len, F), dtype=arr.dtype)], axis=1)
    idx = np.arange(N)[:, None] + seq_len + np.arange(pred_len)[None]
    y = arr[idx, target_idx][..., None]
    return (np.ascontiguousarray(enc, dtype="float32"), np.ascontiguousarray(dec, dtype="float32"),
            y.astype("float32"))


def metricas(y, p):
    e = p - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {"RMSE": float(np.sqrt(np.mean(e ** 2))), "MAE": float(np.mean(np.abs(e))),
            "R2": float(1 - np.sum(e ** 2) / ss_tot) if ss_tot > 0 else float("nan"), "n": int(len(y))}


def block_bootstrap_delta(e_a, e_b, block=24, n_boot=2000, seed=0, stat="rmse"):
    """Diferencia pareada stat(A)-stat(B) sobre las MISMAS horas, remuestreando bloques de `block` horas
    (conserva la autocorrelación). Devuelve (delta, IC95_inf, IC95_sup, prop_boot_<=0 / prop_boot_>=0 mínima*2)."""
    e_a, e_b = np.asarray(e_a, float), np.asarray(e_b, float)
    if e_a.ndim != 1 or e_b.ndim != 1:
        raise ValueError("e_a y e_b deben ser vectores unidimensionales")
    if e_a.shape != e_b.shape:
        raise ValueError("e_a y e_b deben tener la misma longitud para la comparación pareada")
    if block < 1 or len(e_a) < block:
        raise ValueError("block debe ser positivo y no mayor que el número de observaciones")
    if n_boot < 1:
        raise ValueError("n_boot debe ser al menos 1")

    n = len(e_a); nb = int(np.ceil(n / block))
    f = (lambda e: np.sqrt(np.mean(e ** 2, axis=-1))) if stat == "rmse" else (lambda e: np.mean(np.abs(e), axis=-1))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, nb))
    d = np.empty(n_boot, dtype=float)
    offsets = np.arange(block)[None, None, :]
    # Limita la memoria temporal: evita materializar (n_boot, n) índices y
    # dos copias de errores para todas las réplicas simultáneamente.
    chunk_size = max(1, min(64, n_boot))
    for inicio in range(0, n_boot, chunk_size):
        fin = min(inicio + chunk_size, n_boot)
        idx = (starts[inicio:fin, :, None] + offsets).reshape(fin - inicio, -1)[:, :n]
        d[inicio:fin] = f(e_a[idx]) - f(e_b[idx])
    obs = float(f(e_a) - f(e_b))
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = float(2 * min((d <= 0).mean(), (d >= 0).mean()))
    return obs, float(lo), float(hi), p
