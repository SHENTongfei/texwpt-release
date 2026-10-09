# -*- coding: utf-8 -*-
"""TEXWPT S3: MHN blind external probe (R4 dual-external completion).
Protocol (ex-ante honest): MHN = 865.5MHz/other-rectenna/other-lab -> model-level zero-shot impossible
(training domain 2.4GHz). Two legal probes:
 (1) Friis/rectifier physics anchor: Vhat = c*sqrt(P_rx_W * R_load), c calibrated on train fold only.
 (2) Architecture transferability: TEXWPT gated-residual head vs TabPFN/LightGBM/Ridge, within-MHN
     GroupKFold(by distance) x 8 seeds. Relative ranking is the claim; absolute MAE not comparable cross-domain.
Type-IV: MHN measured labels; c calibration on train fold labels only (standard anchor calibration).
"""
import os, json, time, random
os.environ.setdefault("OMP_NUM_THREADS", "6")
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.io import loadmat
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7, 123, 99, 555, 777, 31337]
BASE_DIR = f"{ROOT}/data/extracted/mhn/Wireless_Power_Transfer_MHN-main"

FILES = [
    ("YagiUda", "MH", "DLOS", "PB_and_ESN_YagiUda_antennas", "data2"),
    ("YagiUda", "MH", "NLOS", "PB_and_ESN_YagiUda_antennas", "data2"),
    ("YagiUda", "noMH", "DLOS", "PB_and_ESN_YagiUda_antennas", "data2"),
    ("logper", "MH", "DLOS", "PB_and_ESN_logperiodic_antennas", "data1"),
    ("logper", "noMH", "DLOS", "PB_and_ESN_logperiodic_antennas", "data1"),
]
rows = []
for ant, metal, chan, sub, pref in FILES:
    d = loadmat(f"{BASE_DIR}/{sub}/data/{pref}_{metal}_{chan}_ESN_rectified_voltage.mat")
    V = np.asarray(d["V_rms_all"]).ravel()
    dist = np.asarray(d["distance"]).ravel()
    R = float(np.asarray(d["resistor_load"]).ravel()[0]) if "resistor_load" in d else 3000.0
    dp = f"{BASE_DIR}/{sub}/data/{pref}_{metal}_{chan}_ESN_received_power.mat"
    try:
        P = np.asarray(loadmat(dp)["Power_ESN_rx_av_all_cal_dBm"]).ravel()
    except Exception:
        P = np.full(len(V), np.nan)
    for i in range(len(V)):
        rows.append(dict(ant=ant, metal=metal, chan=chan, dist_m=float(dist[i]),
                         P_rx_dBm=float(P[i]) if np.isfinite(P[i]) else np.nan,
                         R_load=R, V=float(V[i])))
df = pd.DataFrame(rows)
df["pin_lin_W"] = 10 ** ((df.P_rx_dBm - 30) / 10)
df["anchor_phys"] = np.sqrt(df.pin_lin_W * df.R_load)  # V=sqrt(P*R) shape, c calibrated in-fold
df["cond"] = df.ant + "_" + df.metal + "_" + df.chan
print(f"MHN probe rows={len(df)} conds={df.cond.nunique()} P_rx range={df.P_rx_dBm.min():.1f}~{df.P_rx_dBm.max():.1f} dBm")

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

# ---- (1) physics anchor, c in-fold ----
def anchor_pred(tr, te):
    c = float((tr.V / tr.anchor_phys).median())
    return c * te.anchor_phys.values

# ---- (2) gated-residual architecture (TEXWPT head), within-MHN ----
class GateNet(nn.Module):
    def __init__(self, n_feat, d=32, L=1):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(n_feat, d)
        self.emb = nn.Parameter(torch.randn(n_feat, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 2, 64, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1))
        self.gate = nn.Sequential(nn.Linear(d, 8), nn.GELU(), nn.Linear(8, 1), nn.Sigmoid())
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

FEATS = ["P_rx_dBm", "dist_m"]
CATS = {"ant": ["YagiUda", "logper"], "metal": ["MH", "noMH"], "chan": ["DLOS", "NLOS"]}
NTOK = len(FEATS) + 3  # ant, metal, chan
def tok_matrix(d):
    xs = [d[c].fillna(d[c].median() if d[c].notna().any() else 0.0).values.astype(np.float32) for c in FEATS]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats if cat in set(d[k])]
    return np.stack(xs, axis=1), len(xs)

def gate_pred(seed, tr, te):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    xs_tr, ntok = tok_matrix(tr); xs_te, _ = tok_matrix(te)
    mu = xs_tr.mean(0); sd = np.maximum(xs_tr.std(0), 1e-4)
    Xtr = torch.tensor((xs_tr - mu) / sd, dtype=torch.float32)
    Xte = torch.tensor((xs_te - mu) / sd, dtype=torch.float32)
    tid = torch.arange(ntok).expand(len(tr), -1); tidt = torch.arange(ntok).expand(len(te), -1)
    a_tr = torch.tensor(tr.anchor_phys.values, dtype=torch.float32)
    a_te = torch.tensor(te.anchor_phys.values, dtype=torch.float32)
    c = float((tr.V / tr.anchor_phys).median())
    a_tr = a_tr * c; a_te = a_te * c
    ytr = torch.tensor(tr.V.values, dtype=torch.float32)
    net = GateNet(ntok)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    for ep in range(250):
        net.train(); h = net(Xtr, tid)
        pred = a_tr + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
        loss = ((pred - ytr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h = net(Xte, tidt)
        return (a_te + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)).numpy()

def tab_feats(d):
    x = d[FEATS].fillna(0.0).copy()
    x.loc[:, "ant_lp"] = (d.ant == "logper").astype(int).values
    x.loc[:, "metal_mh"] = (d.metal == "MH").astype(int).values
    x.loc[:, "chan_nlos"] = (d.chan == "NLOS").astype(int).values
    return x

from tabpfn import TabPFNRegressor
from lightgbm import LGBMRegressor
def tabpfn_pred(seed, tr, te):
    m = TabPFNRegressor(model_path=f"{ROOT}/models/tabpfn_v2reg/tabpfn-v2-regressor.ckpt", device="cpu")
    m.fit(tab_feats(tr).values, tr.V.values)
    return m.predict(tab_feats(te).values)
def lgbm_pred(seed, tr, te):
    from lightgbm import LGBMRegressor
    m = LGBMRegressor(n_estimators=300, num_leaves=8, learning_rate=0.05, random_state=seed, n_jobs=6, verbose=-1)
    m.fit(tab_feats(tr).values, tr.V.values)
    return m.predict(tab_feats(te).values)
def ridge_pred(seed, tr, te):
    m = Ridge(alpha=1.0)
    sc = StandardScaler().fit(tab_feats(tr))
    m.fit(sc.transform(tab_feats(tr).values), tr.V.values)
    return m.predict(sc.transform(tab_feats(te).values))

MODELS = {"PhysAnchor": lambda s, tr, te: anchor_pred(tr, te),
          "TEXWPT-gate": gate_pred,
          "TabPFN-v2": tabpfn_pred, "LightGBM": lgbm_pred, "Ridge": ridge_pred}
res = {}
t0 = time.time()
for name, fn in MODELS.items():
    ti_folds, ex = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(df, groups=df.dist_m.round(2)):
            tr, te = df.iloc[tr_i], df.iloc[te_i]
            errs.append(mae(fn(seed, tr, te), te.V.values))
        ti_folds.append(errs)
        # blind-shot: train on 80% random rows, test rest (within-MHN repeated holdout as EX analog)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(df)); ntr = int(len(df) * 0.8)
        ex.append(mae(fn(seed, df.iloc[idx[:ntr]], df.iloc[idx[ntr:]]), df.iloc[idx[ntr:]].V.values))
    r = (float(np.mean([np.mean(f) for f in ti_folds])), float(np.std([np.mean(f) for f in ti_folds])),
         float(np.mean(ex)), float(np.std(ex)))
    res[name] = r
    print(f"[{name}] CV={r[0]:.4f}±{r[1]:.4f}  Holdout={r[2]:.4f}±{r[3]:.4f} ({time.time()-t0:.0f}s)", flush=True)

rep = ["# S3 MHN blind external probe (R4)", f"rows={len(df)} conds={df.cond.nunique()}",
       "口径披露: 865.5MHz/他整流器/他实验室 -> 模型级零样本不可行(训练域2.4GHz); 本探针=物理锚对照+架构可迁移性(MHN内部CV/holdout)", "",
       "| 模型 | MHN内CV MAE | 20% holdout MAE |", "|---|---|---|"]
for n, (a, b, c, d) in sorted(res.items(), key=lambda kv: kv[1][2]):
    rep.append(f"| {n} | {a:.4f}±{b:.4f} | {c:.4f}±{d:.4f} |")
out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/s3_mhn_probe.md", "w", encoding="utf-8").write(out)
json.dump({k: list(v) for k, v in res.items()}, open(f"{ROOT}/data/qa/s3_mhn.json", "w"), indent=1)
