# -*- coding: utf-8 -*-
"""KAN baseline only (pykan new API), 8 seeds -> verdict_others.json merge."""
import os, json, time, random
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, torch
from sklearn.model_selection import GroupKFold
from features import d81, d18, mae
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
from kan import KAN as PyKAN
XD = 9
def kan_fit(seed, tr, te):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    m = PyKAN(width=[XD, 8, 1], grid=5, k=3, seed=seed, device="cpu")
    Xt = tab(tr).values
    yt = tr["Y_VDC_V"].values
    # pykan 新版 fit 强制要 test_*; 用训练折内部 20% 切分当其 eval 集 (不接触真实 te, 防早停泄漏)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xt))
    n_val = max(4, int(len(Xt) * 0.2))
    vi, ti_ = idx[:n_val], idx[n_val:]
    Xtr = torch.tensor(Xt[ti_], dtype=torch.float32)
    Xva = torch.tensor(Xt[vi], dtype=torch.float32)
    Xte = torch.tensor(tab(te).values, dtype=torch.float32)
    ytr = torch.tensor(yt[ti_], dtype=torch.float32).reshape(-1, 1)
    yva = torch.tensor(yt[vi], dtype=torch.float32).reshape(-1, 1)
    ds = {"train_input": Xtr, "train_label": ytr, "test_input": Xva, "test_label": yva}
    m.fit(ds, opt="Adam", steps=60, lr=2e-3, loss_fn=torch.nn.MSELoss(), display_metrics=["train_loss"])
    with torch.no_grad():
        return m(Xte).reshape(-1).numpy()
out = json.load(open(f"{ROOT}/data/qa/verdict_others.json"))
t0 = time.time()
ti_folds, ex = [], []
for seed in SEEDS:
    gkf = GroupKFold(n_splits=5); errs = []
    for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
        tr, te = d81.iloc[tr_i], d81.iloc[te_i]
        errs.append(mae(kan_fit(seed, tr, te), te["Y_VDC_V"].values))
    ti_folds.append(errs)
    ex.append(mae(kan_fit(seed, d81, d18), d18["Y_VDC_V"].values))
    print(f"[KAN] seed{seed}: TI={np.mean(errs):.4f} EX={ex[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
out["KAN(pykan)"] = {"TI_folds": ti_folds, "EX": ex}
json.dump(out, open(f"{ROOT}/data/qa/verdict_others.json", "w"), indent=1)
print("KAN DONE")
