# -*- coding: utf-8 -*-
"""Official pytorch-tabnet baseline (anti-'weak-lite' challenge), 8 seeds -> verdict_others.json."""
import os, json, time, random
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, torch
from sklearn.model_selection import GroupKFold
from features import d81, d18, mae
from pytorch_tabnet.tab_model import TabNetRegressor

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7, 123, 99, 555, 777, 31337]
NUMC = ["f_GHz", "pin_avail", "pin_off_db", "anchor_V", "anchor_dB"]
def tab(d):
    x = d[NUMC].fillna(0.0).copy()
    x.loc[:, "mech_b"] = (d.mech == "bent").astype(int).values
    x.loc[:, "proto_r15"] = (d.proto == "r15").astype(int).values
    x.loc[:, "proto_r20"] = (d.proto == "r20").astype(int).values
    x.loc[:, "coup_body"] = (d.coup == "body").astype(int).values
    return x
out = json.load(open(f"{ROOT}/data/qa/verdict_others.json"))
t0 = time.time()
ti_folds, ex = [], []
for seed in SEEDS:
    gkf = GroupKFold(n_splits=5); errs = []
    for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
        tr, te = d81.iloc[tr_i], d81.iloc[te_i]
        torch.manual_seed(seed)
        m = TabNetRegressor(seed=seed, verbose=0, device_name="cpu")
        m.fit(tab(tr).values, tr[["Y_VDC_V"]].values, max_epochs=300, patience=40, batch_size=64)
        errs.append(mae(m.predict(tab(te).values).ravel(), te.Y_VDC_V.values))
    ti_folds.append(errs)
    torch.manual_seed(seed)
    m = TabNetRegressor(seed=seed, verbose=0, device_name="cpu")
    m.fit(tab(d81).values, d81[["Y_VDC_V"]].values, max_epochs=300, patience=40, batch_size=64)
    ex.append(mae(m.predict(tab(d18).values).ravel(), d18.Y_VDC_V.values))
    print(f"[TabNet-official] seed{seed}: TI={np.mean(errs):.4f} EX={ex[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
out["TabNet(official)"] = {"TI_folds": ti_folds, "EX": ex}
json.dump(out, open(f"{ROOT}/data/qa/verdict_others.json", "w"), indent=1)
print("TABNET OFFICIAL DONE")
