import os, sys, json, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()))
from final_data import *
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

def localizar(nombre):
    cwd = Path.cwd().resolve()
    for base in [cwd, *cwd.parents]:
        for sub in ["", "modelos y bd", "metadatos", "data"]:
            p = base / sub / nombre
            if p.is_file(): return p
    raise FileNotFoundError(nombre)

SEQ, LABEL, PRED = 24, 12, 1
OUT = Path("resultados_final_enmascarado"); OUT.mkdir(exist_ok=True)
df = load_bogota(localizar("bogota_calidad_aire_raw.parquet"), estacion="FONTIBÓN")
d = prepare_v2(df, "PM2.5", "2024-01-01", seq_len=SEQ)
masks = {"train": d.obs_train,
         "val": np.r_[d.obs_train[-SEQ:], d.obs_val],
         "test": np.r_[d.obs_val[-SEQ:], d.obs_test]}
W = {k: make_windows(a, SEQ, LABEL, PRED, d.target_idx, masks[k]) for k, a in
     [("train", d.train), ("val", d.val_ctx), ("test", d.test_ctx)]}
flat = lambda k: W[k][0].reshape(len(W[k][0]), -1)
mtr = W["train"][3][:, 0, 0]
Xtr_all, ytr_all = flat("train"), W["train"][2][:, 0, 0]
Xtr, ytr = Xtr_all[mtr], ytr_all[mtr]
Xva, Xte = flat("val"), flat("test")
assert np.array_equal(W["test"][3][:, 0, 0], d.obs_test)
assert int(d.obs_test.sum()) == 8547
y_test, y_val = d.inverse_target(W["test"][2][:, 0, 0]), d.inverse_target(W["val"][2][:, 0, 0])

F_T = OUT / "base_tiempos.json"
tiempos = json.loads(F_T.read_text()) if F_T.exists() else {}
def guardar(nombre, p_te, p_va, s_tr, s_in):
    np.savez_compressed(OUT / f"{nombre}_s0_preds.npz", pred_test=d.inverse_target(p_te), true_test=y_test,
                        obs_test=d.obs_test, pred_val=d.inverse_target(p_va), true_val=y_val, obs_val=d.obs_val,
                        test_timestamps=d.test_index.astype("datetime64[ns]").astype(str),
                        val_timestamps=d.val_index.astype("datetime64[ns]").astype(str))
    tiempos[nombre] = [s_tr, s_in]; F_T.write_text(json.dumps(tiempos))
    m = metricas(y_test[d.obs_test], d.inverse_target(p_te)[d.obs_test])
    print(nombre, {k: round(v, 4) for k, v in m.items()}, flush=True)

ti = d.target_idx
if "persistencia" not in tiempos:
    guardar("persistencia", W["test"][0][:, -1, ti], W["val"][0][:, -1, ti], 0.0, 0.0)
if "ridge" not in tiempos:
    t0 = time.perf_counter(); r = Ridge(alpha=1.0).fit(Xtr, ytr); a = time.perf_counter() - t0
    t0 = time.perf_counter(); p = r.predict(Xte); b = time.perf_counter() - t0
    guardar("ridge", p, r.predict(Xva), a, b)
if "rf" not in tiempos:
    t0 = time.perf_counter()
    rf = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, max_features=0.2, n_jobs=2, random_state=0).fit(Xtr, ytr)
    a = time.perf_counter() - t0
    t0 = time.perf_counter(); p = rf.predict(Xte); b = time.perf_counter() - t0
    guardar("rf", p, rf.predict(Xva), a, b)
print("listo")