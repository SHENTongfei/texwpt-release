# -*- coding: utf-8 -*-
"""TEXWPT G1-v2: 口径统一. 81813 Pin=整流器输入口径(readme), 81785 用 Friis 派生 pin_avail_est
统一到整流器口径; PARD 锚特征; 真纯锚管线对照(Friis+HFSS gain+ADS 整流曲面, 零实测拟合); RF 同口径.
Type-IV: 标签 100% 实测; ADS/HFSS 仅特征与对照.
"""
import os, random
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv"); df["pool"] = df["pool"].astype(str)
an = pd.read_csv(f"{ROOT}/data/processed/anchors.csv")

g81 = an[an.anchor_id == "81785_Fig4c_gain_air"].sort_values("x1")
G81F, G81V = g81.x1.values, g81.y_val.values
ads18v = an[an.anchor_id == "81813_Fig5_ADS_VDC"]
z18g = an[an.anchor_id == "81813_Fig6_HFSS_Z"]
Z18R = z18g[z18g.x2 == 0].sort_values("x1"); Z18I = z18g[z18g.x2 == 1].sort_values("x1")

lam = 0.125
PG = 20 * np.log10(lam / (4 * np.pi * df["dist_m"].clip(lower=0.05).fillna(0.5)))
gr81 = np.interp(df["f_GHz"], G81F, G81V)
df["pin_avail"] = np.where(df.pool == "81785", 27.0 + gr81 + PG, df["Pin_dBm"])
# Fig13 rot 行无 Pin: 排除出回归(EX 腿已排除)
df.loc[df.pool == "81813", "gr_hfss"] = np.nan

def nn2(ads, R, X):
    d = (ads.x1.values - R) ** 2 + (ads.x2.values - X) ** 2
    return float(ads.y_val.values[np.argmin(d)])

zr18 = np.interp(df["f_GHz"], Z18R.x1.values, Z18R.y_val.values)
zi18 = np.interp(df["f_GHz"], Z18I.x1.values, Z18I.y_val.values)
df["ads_VDC"] = [nn2(ads18v, a, b) for a, b in zip(zr18, zi18)]
df["pin_off_db"] = df["pin_avail"] - (-10.0)
# ADS 锚 pin 外推(物理近似: 电压 ∝ 10^(ΔdB/20), ex-ante 声明)
df["anchor_VDC_pin"] = df["ads_VDC"] * (10 ** (df["pin_off_db"] / 20))

COLS = ["f_GHz", "pin_avail", "df_res_GHz", "anchor_VDC_pin", "pin_off_db"]
FILL = {"f_GHz": 2.4, "pin_avail": -10.0, "df_res_GHz": 0.0, "anchor_VDC_pin": 0.15, "pin_off_db": 0.0}
CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
NC = len(COLS) + sum(len(v) for v in CATS.values())

def row_tokens(d):
    xs = [d[c].fillna(FILL[c]).values.astype(np.float32) for c in COLS]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)

def fit_sc(d):
    return {c: (d[c].fillna(FILL[c]).mean(), max(d[c].fillna(FILL[c]).std(), 1e-4)) for c in COLS}

def scale(sc, xs):
    o = xs.copy()
    for j, c in enumerate(COLS):
        mu, sd = sc[c]; o[:, j] = (o[:, j] - mu) / sd
    return o

class Net(nn.Module):
    def __init__(self, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(NC, d)
        self.emb = nn.Parameter(torch.randn(NC, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.g = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
        self.ab = nn.Parameter(torch.zeros(1))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)
    def reg(self, p): return self.g(p).squeeze(-1) + self.ab
    def proj(self, p): return nn.functional.normalize(self.pj(p), dim=-1)

def run(seed, tr_df, te_df):
    import torch as _t
    _t.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    sc = fit_sc(tr_df)
    Xtr = _t.tensor(scale(sc, row_tokens(tr_df)), dtype=_t.float32, device=DEV)
    Xte = _t.tensor(scale(sc, row_tokens(te_df)), dtype=_t.float32, device=DEV)
    tid = _t.arange(NC, device=DEV).expand(len(tr_df), -1); tidt = _t.arange(NC, device=DEV).expand(len(te_df), -1)
    y = _t.tensor(tr_df["Y_VDC_V"].values.astype(np.float32), device=DEV)
    net = Net().to(DEV); net.ab.data.fill_(float(tr_df.Y_VDC_V.mean()))
    opt = _t.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(100):
        net.train(); m = (_t.rand(Xtr.shape, device=DEV) < 0.3); m[:, 3:] = False
        Xi = Xtr.clone(); Xi[m] = 0.0
        loss = ((net(Xi, tid) - net(_t.where(m, Xtr, Xi), tid)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    grp = (tr_df["mech"].astype(str) + tr_df["proto"].astype(str)).astype("category").cat.codes.values.astype(np.int64)
    yt = _t.tensor(grp, device=DEV)
    for ep in range(60):
        net.train(); p = net.proj(net(Xtr, tid))
        mask = (yt[:, None] == yt[None, :]) & (~_t.eye(len(yt), dtype=_t.bool, device=DEV))
        if mask.sum() == 0: break
        sim = (p @ p.T / 0.07).masked_fill(_t.eye(len(yt), dtype=_t.bool, device=DEV), -1e9)
        loss = nn.functional.cross_entropy(sim, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    for ep in range(300):
        net.train(); loss = ((net.reg(net(Xtr, tid)) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with _t.no_grad(): pr = net.reg(net(Xte, tidt)).cpu().numpy()
    return np.clip(pr, 0.01, 2.0)

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
rep = ["# G1 v2 report (统一整流器口径 + 真纯锚 + RF 同口径)", f"dev={DEV}", ""]

d81 = df[(df.pool == "81785") & df.Y_VDC_V.notna()].copy()
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()].copy()

# ---- 真纯锚管线 (81813 侧, 零实测拟合): VDC = ADS(Fig5)@Zant(f) × 10^(ΔdB/20) ----
pa = d18.anchor_VDC_pin.values
rep.append(f"PureAnchor(81813, Friis口径ADS外推): MAE_V = {mae(pa, d18.Y_VDC_V.values):.4f}")

# ---- RF 同口径 ----
FE = ["f_GHz", "pin_avail", "df_res_GHz", "anchor_VDC_pin", "pin_off_db",
      "mech_b", "proto_r15", "proto_r20"]
def fer(d):
    x = pd.DataFrame(index=d.index)
    for c in FE[:5]: x[c] = d[c].fillna(0.0)
    x["mech_b"] = (d.mech == "bent").astype(int)
    x["proto_r15"] = (d.proto == "r15").astype(int); x["proto_r20"] = (d.proto == "r20").astype(int)
    return x
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=8)
rf.fit(fer(d81), d81.Y_VDC_V)
rep.append(f"RF(同口径) EX: MAE_V = {mae(rf.predict(fer(d18)), d18.Y_VDC_V.values):.4f}")
gkf = GroupKFold(n_splits=5); errs = []
for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
    rf2 = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=8)
    rf2.fit(fer(d81.iloc[tr_i]), d81.iloc[tr_i].Y_VDC_V)
    errs.append(mae(rf2.predict(fer(d81.iloc[te_i])), d81.iloc[te_i].Y_VDC_V))
rep.append(f"RF(同口径) TI: MAE_V = {np.mean(errs):.4f}±{np.std(errs):.4f}")

# ---- v2 net, 5 seeds ----
ti_all, ex_all = [], []
for seed in [42, 2024, 7, 123, 99]:
    gkf = GroupKFold(n_splits=5); errs = []
    for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
        pr = run(seed, d81.iloc[tr_i], d81.iloc[te_i])
        errs.append(mae(pr, d81.iloc[te_i].Y_VDC_V.values))
    ti = np.mean(errs)
    pr = run(seed, d81, d18); ex = mae(pr, d18.Y_VDC_V.values)
    ti_all.append(ti); ex_all.append(ex)
    rep.append(f"seed{seed}: TI={ti:.4f}  EX={ex:.4f}")
rep.append(f"\nSUM TI v2: {np.mean(ti_all):.4f}±{np.std(ti_all):.4f}")
rep.append(f"SUM EX v2: {np.mean(ex_all):.4f}±{np.std(ex_all):.4f}   (RF 同口径见上; v1: 0.323±0.049)")
out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/g1_v2_report.md", "w", encoding="utf-8").write(out)
