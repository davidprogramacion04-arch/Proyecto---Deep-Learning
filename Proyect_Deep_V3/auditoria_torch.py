"""Auditoría del STN (Entrega final §8.2) — script de TERMINAL, sin matplotlib ni sklearn.
Uso (Anaconda Prompt, ambiente activado, en la carpeta donde están final_data.py y final_stn.py):
    python auditoria_torch.py
Lee resultados_final/modelos/*.pt y escribe resultados_final/auditoria.json y auditoria_atencion.npz"""
import os
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[v] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from final_data import *
import final_stn as S
DEVICE = S.DEVICE

SEQ, LABEL, PRED = 24, 12, 1
CFG = dict(d_model=128, n_heads=8, d_ff=256, dropout=0.05, factor=5, e_layers=3, d_layers=2, seq_len=SEQ, label_len=LABEL)
AUDIT = [("scratch", 0), ("ft_full", 0)]
REP = 3
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "resultados_final")

def localizar(nombre):
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        for sub in ["", "modelos y bd", "metadatos", "data"]:
            p = base / sub / nombre
            if p.is_file(): return p
    raise FileNotFoundError(nombre)

def log(*a): print(*a, flush=True)

log("dispositivo:", DEVICE, "| torch", torch.__version__)
df = load_bogota(localizar("bogota_calidad_aire_raw.parquet"), estacion="FONTIBÓN")
d = prepare_v2(df, "PM2.5", "2024-01-01", seq_len=SEQ)
e_np, d_np, y_np = make_windows(d.test_ctx, SEQ, LABEL, PRED, d.target_idx)
ENC, DEC = torch.tensor(e_np, device=DEVICE), torch.tensor(d_np, device=DEVICE)
names, ti = d.feature_names, d.target_idx
rg_t = float(d.scaler_max[ti] - d.scaler_min[ti])
y_test = d.inverse_target(y_np[:, 0, 0]); obs = d.obs_test
assert len(ENC) == len(obs)
log("ventanas de prueba:", len(ENC), "| observadas:", int(obs.sum()))
rng_t = torch.Generator(device="cpu"); rng_t.manual_seed(123)

def cargar(r, s):
    m = S.construir_modelo(d.train.shape[1], CFG, PRED)
    m.load_state_dict(torch.load(OUT / "modelos" / f"stn_{r}_s{s}.pt", map_location=DEVICE)); return m.eval()

def pred(model, enc, dec): return d.inverse_target(S.predecir(model, enc, dec, S.fwd_stn))
def rmse(p): return float(np.sqrt(np.mean((p[obs] - y_test[obs]) ** 2)))

def permutar(enc, dec, cols, lags):
    outs = []
    for _ in range(REP):
        perm = torch.randperm(len(enc), generator=rng_t).to(DEVICE)
        e, dd = enc.clone(), dec.clone()
        for s in lags:
            e[:, s, cols] = enc[perm][:, s, cols]
            r = s - (SEQ - LABEL)
            if r >= 0: dd[:, r, cols] = dec[perm][:, r, cols]
        outs.append((e, dd))
    return outs

def contrafactual(model, p0, var, delta_real, solo_ultimo=False):
    j = names.index(var); rg = float(d.scaler_max[j] - d.scaler_min[j])
    e, dd = ENC.clone(), DEC.clone()
    if solo_ultimo: e[:, -1, j] += delta_real / rg; dd[:, LABEL - 1, j] += delta_real / rg
    else: e[:, :, j] += delta_real / rg; dd[:, :LABEL, j] += delta_real / rg
    return float((pred(model, e, dd) - p0)[obs].mean())

def atencion(model, idx):
    ia = model.encoder.encoders[0].attn_layers[0].attention.inner_attention
    ia.capture = True
    with torch.no_grad(): model(ENC[idx], DEC[idx])
    ia.capture = False
    a, ind = ia.last_attn, ia.last_index
    B, H, u, L = a.shape
    A = torch.full((B, H, L, L), 1.0 / L)
    A.scatter_(2, ind.unsqueeze(-1).expand(-1, -1, -1, L), a)
    return A.mean(dim=(0, 1)).numpy(), (int(u), int(L))

pm_ult = d.inverse_target(e_np[:, -1, ti])
grupos = {"bajo_p25": np.where(pm_ult < np.percentile(pm_ult, 25))[0], "alto_p90": np.where(pm_ult > np.percentile(pm_ult, 90))[0]}

res, att = {}, {}
for r, s in AUDIT:
    t0 = time.time(); log(f"\n== {r} s{s} ==")
    m = cargar(r, s); p0 = pred(m, ENC, DEC); base = rmse(p0)
    log("RMSE base:", round(base, 4))
    imp_var = {}
    for j, n in enumerate(names):
        imp_var[n] = float(np.mean([rmse(pred(m, e, dd)) - base for e, dd in permutar(ENC, DEC, [j], range(SEQ))]))
        log("  variable", n, round(imp_var[n], 4))
    imp_lag = {}
    for k in range(SEQ):
        imp_lag[str(SEQ - k)] = float(np.mean([rmse(pred(m, e, dd)) - base for e, dd in permutar(ENC, DEC, [ti], [k])]))
    log("  retardo 1 h:", round(imp_lag["1"], 4), "| retardo 24 h:", round(imp_lag["24"], 4))
    reps = np.array([S.predecir(m, ENC, DEC, S.fwd_stn) for _ in range(5)])
    sd_inf = float(reps.std(axis=0).mean() * rg_t)
    cf = {"PM2.5_ultima_hora_+10ug": contrafactual(m, p0, "PM2.5", 10.0, solo_ultimo=True)}
    for v in names:
        if v != "PM2.5": cf[f"{v}_+1sd"] = contrafactual(m, p0, v, float(d.train_std[names.index(v)]))
    for g, ids in grupos.items():
        A, (u, L) = atencion(m, torch.tensor(ids[:512], device=DEVICE)); att[f"{r}_s{s}|{g}"] = A
    res[f"{r}_s{s}"] = {"rmse_base": base, "imp_var": imp_var, "imp_lag": imp_lag, "sd_inferencia_ugm3": sd_inf,
                        "contrafactual": cf, "atencion_u": u, "atencion_L": L}
    (OUT / "auditoria.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    np.savez_compressed(OUT / "auditoria_atencion.npz", **att)
    log(f"  sd por muestreo de keys: {sd_inf:.4f} µg/m³ | contrafactual PM2.5 +10: {cf['PM2.5_ultima_hora_+10ug']:.3f} | {time.time()-t0:.0f}s")
log("\nListo → resultados_final/auditoria.json y auditoria_atencion.npz")
