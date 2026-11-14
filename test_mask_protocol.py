"""Prueba rápida del protocolo de máscara con una serie sintética."""
import numpy as np
import pandas as pd
from final_data import prepare_v2, make_windows

def main():
    idx = pd.date_range("2020-01-01", periods=1000, freq="h")
    df = pd.DataFrame({
        "PM2.5": np.linspace(5, 50, len(idx)),
        "met": np.sin(np.arange(len(idx)) / 10),
    }, index=idx)
    df.loc[idx[[3, 30, 90, 180]], "PM2.5"] = np.nan
    d = prepare_v2(df, "PM2.5", "2020-01-30", seq_len=24)
    masks = {
        "train": d.obs_train,
        "val": np.r_[d.obs_train[-24:], d.obs_val],
        "test": np.r_[d.obs_val[-24:], d.obs_test],
    }
    w_val = make_windows(d.val_ctx, 24, 12, 1, d.target_idx, masks["val"])
    w_test = make_windows(d.test_ctx, 24, 12, 1, d.target_idx, masks["test"])
    assert np.array_equal(w_val[3][:, 0, 0], d.obs_val)
    assert np.array_equal(w_test[3][:, 0, 0], d.obs_test)
    assert w_test[0].shape[0] == len(d.obs_test)
    print("OK: la máscara excluye etiquetas imputadas y conserva las ventanas de entrada.")

if __name__ == "__main__":
    main()
