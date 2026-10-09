# -*- coding: utf-8 -*-
"""TEXWPT verdict (others): TabPFN/GP/SVR/KNN/Ridge, 8 seeds, per-fold TI + per-seed EX -> json."""
import os, json, time
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from features import d81, d18, mae

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7, 123, 99, 555, 777, 31337]
NUMC = ["f_GHz", "pin_avail", "pin_off_db", "anchor_V", "anchor_dB"]

def tab(d):
    x = d[NUMC].fillna(0.0).copy()
    x = x.copy()
    x.loc[:, "mech_b"] = (d.mech == "bent").astype(int).values
    x.loc[:, "proto_r15"] = (d.proto == "r15").astype(int).values
    x.loc[:, "proto_r20"] = (d.proto == "r20").astype(int).values
    x.loc[:, "coup_body"] = (d.coup == "body").astype(int).values
    return x

from tabpfn import TabPFNRegressor
CK = f"{ROOT}/models/tabpfn_v2reg/tabpfn-v2-regressor.ckpt"

def fit_predict_tabpfn(tr, te):
    m = TabPFNRegressor(model_path=CK, device="cpu")
    m.fit(tab(tr).values, tr.Y_VDC_V.values)
    return m.predict(tab(te).values)

def fit_predict_sk(maker, seed, tr, te):
    m = maker(seed)
    sc = StandardScaler().fit(tab(tr))
    m.fit(sc.transform(tab(tr).values), tr.Y_VDC_V.values)
    return m.predict(sc.transform(tab(te).values))

MAKERS = {
    "TabPFN-v2": None,
    "GP": lambda s: GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0), normalize_y=True, random_state=s),
    "SVR": lambda s: SVR(C=3.0, epsilon=0.01),
    "KNN": lambda s: KNeighborsRegressor(5),
    "Ridge": lambda s: Ridge(alpha=1.0),
}
out_p = f"{ROOT}/data/qa/verdict_others.json"
try:
    out = json.load(open(out_p))
except Exception:
    out = {}
t0 = time.time()
for name, maker in MAKERS.items():
    ti_folds, ex = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5)
        errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            tr, te = d81.iloc[tr_i], d81.iloc[te_i]
            pred = fit_predict_tabpfn(tr, te) if maker is None else fit_predict_sk(maker, seed, tr, te)
            errs.append(mae(pred, te.Y_VDC_V.values))
        ti_folds.append(errs)
        pred = fit_predict_tabpfn(d81, d18) if maker is None else fit_predict_sk(maker, seed, d81, d18)
        ex.append(mae(pred, d18.Y_VDC_V.values))
        print(f"[{name}] seed{seed}: TI={np.mean(errs):.4f} EX={ex[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    out[name] = {"TI_folds": ti_folds, "EX": ex}
    json.dump(out, open(out_p, "w"), indent=1)
json.dump(out, open(out_p, "w"), indent=1)
print("OTHERS DONE")
