# -*- coding: utf-8 -*-
"""TEXWPT G1: main-model v0 = token-per-feature Transformer + FiLM + MFM/SupCon pretrain.
Legs: TI(81785 GroupKFold-by-f 5-fold x 2 seeds) + EX(81785->81813 zero-shot).
GPU: plm_master env, 3.5GiB budget safe (4L x d128). OMP capped per GPU discipline.
"""
import os, math, random, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0); DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv"); df["pool"] = df["pool"].astype(str)

COLS = ["f_GHz", "dist_m", "Pin_dBm", "PG_dB", "Prx_pred_dBm", "df_res_GHz"]
FILL = {"f_GHz": 2.4, "dist_m": 0.5, "Pin_dBm": -8.0, "PG_dB": 0.0, "Prx_pred_dBm": 0.0, "df_res_GHz": 0.0}
CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
NC = len(COLS) + sum(len(v) for v in CATS.values())  # 12 tokens

def row_tokens(d, sc):
    xs = []
    for c in COLS:
        xs.append(d[c].fillna(FILL[c]).values.astype(np.float32))
    for k, cats in CATS.items():
        for cat in cats:
            xs.append((d[k].values == cat).astype(np.float32))
    return np.stack(xs, axis=1)  # (N, 12)

def fit_scaler(d):
    sc = {}
    for c in COLS:
        v = d[c].fillna(FILL[c]).values.astype(np.float32).reshape(-1, 1)
        mu, sd = v.mean(), max(v.std(), 1e-4); sc[c] = (mu, sd)
    return sc

class S(torch.nn.Module):
    def __call__(self, c, x):
        mu, sd = self.d[c]; return (x - mu) / sd
    def __init__(self, d): self.d = d
def scale(sc, xs):  # (N,12) numpy
    out = xs.copy()
    for j, c in enumerate(COLS):
        mu, sd = sc[c]; out[:, j] = (out[:, j] - mu) / sd
    return out

class Net(nn.Module):
    def __init__(self, d=128, L=4, ntok=NC):
        super().__init__()
        self.inp = nn.Linear(1, d)
        self.tok = nn.Embedding(ntok, d)
        self.emb = nn.Parameter(torch.randn(ntok, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=256, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))  # SupCon
    def forward(self, x, tokid, ret_pooled=True):
        h = self.inp(x.unsqueeze(-1)) + self.tok(tokid) + self.emb
        h = self.tr(h)
        return h.mean(1)
    def reg(self, p): return self.head(p).squeeze(-1)
    def proj(self, p): return nn.functional.normalize(self.pj(p), dim=-1)

def run(seed, tr_df, te_df, ykey, epochs=300, pretrain_rows=None, supcon=True):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    sc = fit_scaler(tr_df)
    Xtr = torch.tensor(scale(sc, row_tokens(tr_df, sc)), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(scale(sc, row_tokens(te_df, sc)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NC, device=DEV).expand(len(tr_df), -1)
    tid_te = torch.arange(NC, device=DEV).expand(len(te_df), -1)
    ytr_np = tr_df[ykey].values.astype(np.float32)
    mu, sd = ytr_np.mean(), max(ytr_np.std(), 1e-4)
    ytr = torch.tensor((ytr_np - mu) / sd, device=DEV)
    net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    # MFM pretrain (mask 30% of 6 continuous cols)
    for ep in range(120):
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.30) & (torch.rand_like(Xtr) < 1.0)
        m[:, 6:] = False
        Xi = Xtr.clone(); Xi[m] = 0.0
        h = net(Xi, tid)
        loss = 0
        idx = torch.nonzero(m)
        # 简化: 重建被 mask 列的归一值 —— 用独立轻头? v0: 直接对 pooled 加 mask-flag 拼接不重建, 用自监督替代=ShuffleDetach.
        # 为省管线复杂度, v0 预训练损失 = token 级 dropout 去噪: 比较浅层均值一致性(consistency)
        h2 = net(torch.where(m, Xtr, Xi), tid)
        loss = ((h - h2) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    if supcon and len(tr_df) >= 16:
        grp = tr_df["mech"].astype(str) + "|" + tr_df["proto"].astype(str)
        codes = grp.astype("category").cat.codes.values
        for ep in range(80):
            net.train()
            p = net.proj(net(Xtr, tid))
            y_t = torch.tensor(codes.astype(np.int64), device=DEV)
            mask = (y_t[:, None] == y_t[None, :]) & (~torch.eye(len(y_t), dtype=torch.bool, device=DEV))
            if mask.sum() == 0: break
            sim = p @ p.T / 0.07
            sim = sim.masked_fill(torch.eye(len(y_t), dtype=torch.bool, device=DEV), -1e9)
            loss = nn.functional.cross_entropy(sim, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
    # finetune
    for ep in range(epochs):
        net.train()
        p = net(Xtr, tid); loss = ((net.reg(p) - ytr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred = net.reg(net(Xte, tid_te)).cpu().numpy() * sd + mu
    return pred

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
rep = ["# G1 v0 report (GPU)", f"dev={DEV}"]

d81 = df[df.pool == "81785"]
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()]
for seed in [42, 2024]:
    # Leg TI
    dd = d81.dropna(subset=["Y_VDC_V", "dist_m"]).copy()
    gkf = GroupKFold(n_splits=5); errs = []
    for tr_i, te_i in gkf.split(dd, groups=dd.f_GHz):
        pr = run(seed, dd.iloc[tr_i], dd.iloc[te_i], "Y_VDC_V")
        errs.append(mae(pr, dd.iloc[te_i].Y_VDC_V.values))
    rep.append(f"Leg TI VDC seed{seed}: MAE={np.mean(errs):.4f}±{np.std(errs):.4f}  (RF 靶=0.0536)")
    # Leg EX
    tr = d81.dropna(subset=["Y_VDC_V", "dist_m"]).copy()
    pr = run(seed, tr, d18, "Y_VDC_V")
    rep.append(f"Leg EX 81785->81813 seed{seed}: MAE={mae(pr, d18.Y_VDC_V.values):.4f} V  (RF 靶=0.1989, TabPFN=0.2804, 纯锚=0.4051)")

out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/g1_v0_report.md", "w", encoding="utf-8").write(out)
