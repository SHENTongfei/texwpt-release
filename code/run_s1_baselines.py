# -*- coding: utf-8 -*-
"""TEXWPT S1: named-baseline expansion. LightGBM/CatBoost/XGBoost/RF/TabNet/KAN/FT-Transformer,
8 seeds x (TI 5-fold + EX), per-fold -> verdict_others.json (merge). Protocol identical to run_verdict_others."""
import os, json, time, random
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
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
XD, YC = 9, "Y_VDC_V"
t0 = time.time()

def torch_fit_predict(model_cls, tr, te, seed, epochs=200):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    Xtr = torch.tensor(tab(tr).values, dtype=torch.float32)
    Xte = torch.tensor(tab(te).values, dtype=torch.float32)
    ytr = torch.tensor(tr[YC].values, dtype=torch.float32).reshape(-1, 1)
    net = model_cls()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    lossf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    for ep in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad(); loss = lossf(net(xb), yb); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        return net(Xte).ravel().numpy()

class FTTransformer(nn.Module):
    """FT-Transformer (Gorishniy et al. 2021), feature-tokenizer + 2-layer encoder, self-implemented minimal."""
    def __init__(self, n_feat=XD, d=48, n_head=2, L=2):
        super().__init__()
        self.w = nn.Linear(1, d)
        self.b = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        self.rw = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, n_head, 96, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 1))
    def forward(self, x):
        n, f = x.shape
        tok = self.w(x.unsqueeze(-1)) + self.b.unsqueeze(0) + self.rw.unsqueeze(0)
        cls = self.cls.expand(n, -1, -1)
        h = self.tr(torch.cat([cls, tok], 1))
        return self.head(h[:, 0]).squeeze(-1)

class TabNetLite(nn.Module):
    """Decision-attention block in the spirit of TabNet (Arik & Pfister 2021), minimal learnable mask."""
    def __init__(self, n_feat=XD, d=32):
        super().__init__()
        self.bn = nn.BatchNorm1d(n_feat)
        self.fc1 = nn.Linear(n_feat, 64); self.att = nn.Linear(64, n_feat); self.fc2 = nn.Linear(n_feat, 32)
        self.head = nn.Linear(32, 1)
    def forward(self, x):
        h = torch.relu(self.fc1(self.bn(x)))
        mask = torch.sigmoid(self.att(h))
        h2 = torch.relu(self.fc2(x * mask))
        return self.head(h2).squeeze(-1)

out = json.load(open(f"{ROOT}/data/qa/verdict_others.json"))

def bench(name, fit_pred):
    ti_folds, ex = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            tr, te = d81.iloc[tr_i], d81.iloc[te_i]
            errs.append(mae(fit_pred(seed, tr, te), te[YC].values))
        ti_folds.append(errs)
        ex.append(mae(fit_pred(seed, d81, d18), d18[YC].values))
        print(f"[{name}] seed{seed}: TI={np.mean(errs):.4f} EX={ex[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    out[name] = {"TI_folds": ti_folds, "EX": ex}
    json.dump(out, open(f"{ROOT}/data/qa/verdict_others.json", "w"), indent=1)

def sk(maker):
    def f(seed, tr, te):
        m = maker(seed)
        sc = StandardScaler().fit(tab(tr))
        m.fit(sc.transform(tab(tr).values), tr[YC].values)
        return m.predict(sc.transform(tab(te).values))
    return f

from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
ONLY_KAN = os.environ.get("ONLY_KAN") == "1"
if not ONLY_KAN:
    bench("LightGBM", sk(lambda s: LGBMRegressor(n_estimators=500, num_leaves=15, learning_rate=0.05, random_state=s, n_jobs=6, verbose=-1)))
bench("CatBoost", sk(lambda s: CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05, random_seed=s, verbose=0)))
bench("XGBoost", sk(lambda s: XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=s, n_jobs=6)))
bench("RF", sk(lambda s: RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=6)))
bench("FT-Transformer", lambda s, tr, te: torch_fit_predict(FTTransformer, tr, te, s))
bench("TabNet(tabnet-lite)", lambda s, tr, te: torch_fit_predict(TabNetLite, tr, te, s))

# KAN (pykan)
try:
    from kan import KAN as PyKAN
    def kan_fit(seed, tr, te):
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        m = PyKAN(width=[XD, 8, 1], grid=5, k=3, seed=seed, device="cpu")
        Xtr = torch.tensor(tab(tr).values, dtype=torch.float32)
        Xte = torch.tensor(tab(te).values, dtype=torch.float32)
        ytr = torch.tensor(tr[YC].values, dtype=torch.float32).reshape(-1, 1)
        ds = {"train_input": Xtr, "train_label": ytr}
        m.fit(ds, opt="Adam", steps=60, lr=2e-3, loss_fn=torch.nn.MSELoss(), display_metrics=["train_loss"])
        with torch.no_grad():
            return m(Xte).reshape(-1).numpy()
    bench("KAN(pykan)", kan_fit)
except Exception as e:
    print("KAN skipped:", str(e)[:120])

print("S1 DONE")
