# -*- coding: utf-8 -*-
"""TEXWPT G1-v3: 真 PARD 残差结构 (anchor + g_theta) + I6a/b/c INNO 快筛 (H58).
yhat = anchor_phys(pool) + r_hat; 保底 = anchor (81813 侧 = 0.1242).
臂: base(MLP残差) / +I6a(物理残差基) / +I6b(Smith不变量) / +I6c(Q代理) — 5 seeds x TI/EX.
Type-IV: 标签 100% 实测; ADS/HFSS 仅锚+特征; R_est 从 ADS 仿真侧定标.
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
ads81e = an[an.anchor_id == "81785_Fig2_ADS_eff"]
r3 = an[an.anchor_id == "81813_Fig3rect_ADS_VDC"].merge(
    an[an.anchor_id == "81813_Fig3rect_ADS_eff"], on=["x1"], suffixes=("_v", "_e"))
R_EST = float(np.median((r3.y_val_v ** 2) / (10 ** ((-10 - 30) / 10) * r3.y_val_e)))  # VDC^2/PDC
print(f"R_est from ADS Fig3rect: {R_EST/1000:.2f} kOhm")

z18g = an[an.anchor_id == "81813_Fig6_HFSS_Z"]
Z18R = z18g[z18g.x2 == 0].sort_values("x1"); Z18I = z18g[z18g.x2 == 1].sort_values("x1")
z81g = an[an.anchor_id == "81785_Fig3air_HFSS_Z"]
Z81R = z81g[z81g.x2 == 0].sort_values("x1"); Z81I = z81g[z81g.x2 == 3].sort_values("x1")

lam = 0.125
PG = 20 * np.log10(lam / (4 * np.pi * df["dist_m"].clip(lower=0.05).fillna(0.5)))
gr81 = np.interp(df["f_GHz"], G81F, G81V)
df["pin_avail"] = np.where(df.pool == "81785", 27.0 + gr81 + PG, df["Pin_dBm"])
df["pin_lin_W"] = 10 ** ((df["pin_avail"] - 30) / 10)

def nn2(ads, R, X):
    d = (ads.x1.values - R) ** 2 + (ads.x2.values - X) ** 2
    return float(ads.y_val.values[np.argmin(d)])

zr18 = np.interp(df["f_GHz"], Z18R.x1.values, Z18R.y_val.values)
zi18 = np.interp(df["f_GHz"], Z18I.x1.values, Z18I.y_val.values)
zr81 = np.interp(df["f_GHz"], Z81R.x1.values, Z81R.y_val.values)
zi81 = np.interp(df["f_GHz"], Z81I.x1.values, Z81I.y_val.values)

ads_vdc18 = np.array([nn2(ads18v, a, b) for a, b in zip(zr18, zi18)])
ads_eff81 = np.array([nn2(ads81e, a, b) for a, b in zip(zr81, zi81)])
pin_off = df["pin_avail"].values - (-10.0)
df["pin_off_db"] = pin_off
# anchors
anchor18 = ads_vdc18 * (10 ** (pin_off / 20))
anchor81 = np.sqrt(df["pin_lin_W"].values * ads_eff81 * R_EST)
df["anchor_V"] = np.where(df.pool == "81785", anchor81, anchor18)
df.loc[df.anchor_V <= 0, "anchor_V"] = 0.01

# Smith 不变量 (I6b): 轨迹角 + 归一化模长, 池内 Z_c = f_res 处 Z
for pool, ZR, ZI in [("81785", Z81R, Z81I), ("81813", Z18R, Z18I)]:
    m = df.pool == pool
    fres = df.loc[m, "f_GHz"].iloc[np.argmin(abs(df.loc[m, "f_GHz"].values - 2.4))]
    zc_r = float(np.interp(2.4, ZR.x1.values, ZR.y_val.values))
    zc_i = float(np.interp(2.4, ZI.x1.values, ZI.y_val.values))
    zr = np.interp(df.loc[m, "f_GHz"], ZR.x1.values, ZR.y_val.values)
    zi = np.interp(df.loc[m, "f_GHz"], ZI.x1.values, ZI.y_val.values)
    dz = zr - zc_r + 1j * (zi - zc_i)
    df.loc[m, "smith_ang"] = np.angle(dz)
    df.loc[m, "smith_norm"] = np.abs(dz) / (max(np.abs(dz).max(), 1e-6))
df["smith_ang"] = df["smith_ang"].fillna(0.0); df["smith_norm"] = df["smith_norm"].fillna(0.0)

# Q 代理 (I6c): f_res/FWHM per (pool, mech)
for (pool, mech), g in df.dropna(subset=["Y_VDC_V"]).groupby(["pool", "mech"]):
    gg = g.groupby("f_GHz")["Y_VDC_V"].mean()
    if len(gg) < 5: continue
    fpk = gg.idxmax(); half = gg.max() / 2
    above = gg[gg >= half]
    fwhm = max(above.index.max() - above.index.min(), 0.01)
    df.loc[(df.pool == pool) & (df.mech == mech), "Q_proxy"] = fpk / fwhm
df["Q_proxy"] = df["Q_proxy"].fillna(df["Q_proxy"].median())

# I6a 物理残差基
df["sq_shape"] = 10 ** (df["pin_off_db"] / 20)          # 平方律形状
df["sat_shape"] = 1 / (1 + np.exp(-(df["pin_avail"] + 8) / 2))  # 饱和形状

df["resid_target"] = df["Y_VDC_V"] - df["anchor_V"]
BASE = ["f_GHz", "pin_avail", "pin_off_db", "df_res_GHz"]
ARMS = {
    "base": [],
    "I6a": ["sq_shape", "sat_shape"],
    "I6b": ["smith_ang", "smith_norm"],
    "I6c": ["Q_proxy"],
}
CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}

def feats(arm):
    return BASE + ARMS[arm]

def row_tokens(d, cols):
    FILL = {"f_GHz": 2.4, "pin_avail": -10.0, "pin_off_db": 0.0, "df_res_GHz": 0.0,
            "sq_shape": 1.0, "sat_shape": 0.5, "smith_ang": 0.0, "smith_norm": 0.0, "Q_proxy": 0.0}
    xs = [d[c].fillna(FILL.get(c, 0.0)).values.astype(np.float32) for c in cols]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1), len(cols) + 7

def fit_sc(d, cols):
    FILL = {"f_GHz": 2.4, "pin_avail": -10.0, "pin_off_db": 0.0, "df_res_GHz": 0.0,
            "sq_shape": 1.0, "sat_shape": 0.5, "smith_ang": 0.0, "smith_norm": 0.0, "Q_proxy": 0.0}
    return {c: (d[c].fillna(FILL.get(c, 0.0)).mean(), max(d[c].fillna(FILL.get(c, 0.0)).std(), 1e-4)) for c in cols}

def scale(sc, xs, cols):
    o = xs.copy()
    for j, c in enumerate(cols):
        mu, sd = sc[c]; o[:, j] = (o[:, j] - mu) / sd
    return o

class Net(nn.Module):
    def __init__(self, ntok, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(ntok, d)
        self.emb = nn.Parameter(torch.randn(ntok, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.g = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)
    def proj(self, p): return nn.functional.normalize(self.pj(p), dim=-1)

def run(seed, tr_df, te_df, cols, ntok, epochs=300):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    sc = fit_sc(tr_df, cols)
    Xtr = torch.tensor(scale(sc, row_tokens(tr_df, cols)[0], cols), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(scale(sc, row_tokens(te_df, cols)[0], cols), dtype=torch.float32, device=DEV)
    tid = torch.arange(ntok, device=DEV).expand(len(tr_df), -1)
    tidt = torch.arange(ntok, device=DEV).expand(len(te_df), -1)
    # 残差目标, 标准化
    rtr = tr_df["resid_target"].values.astype(np.float32)
    mu, sd = rtr.mean(), max(rtr.std(), 1e-3)
    y = torch.tensor((rtr - mu) / sd, device=DEV)
    net = Net(ntok).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(100):
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(cols):] = False
        Xi = Xtr.clone(); Xi[m] = 0.0
        loss = ((net(Xi, tid) - net(torch.where(m, Xtr, Xi), tid)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    grp = (tr_df["mech"].astype(str) + tr_df["proto"].astype(str)).astype("category").cat.codes.values.astype(np.int64)
    yt = torch.tensor(grp, device=DEV)
    for ep in range(60):
        net.train(); p = net.proj(net(Xtr, tid))
        mask = (yt[:, None] == yt[None, :]) & (~torch.eye(len(yt), dtype=torch.bool, device=DEV))
        if mask.sum() == 0: break
        sim = (p @ p.T / 0.07).masked_fill(torch.eye(len(yt), dtype=torch.bool, device=DEV), -1e9)
        loss = nn.functional.cross_entropy(sim, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    for ep in range(epochs):
        net.train(); loss = ((net.g(net(Xtr, tid)).squeeze(-1) - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad(): dr = net.g(net(Xte, tidt)).squeeze(-1).cpu().numpy() * sd + mu
    return te_df["anchor_V"].values + dr  # yhat = anchor + residual

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
rep = [f"# G1 v3 + I6 quick screen (H58), R_est={R_EST/1000:.2f}k, dev={DEV}", ""]

d81 = df[(df.pool == "81785") & df.Y_VDC_V.notna()].copy()
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()].copy()
rep.append(f"anchor-alone: 81813 EX = {mae(d18.anchor_V, d18.Y_VDC_V):.4f} (保底) | 81785 anchor TI = {mae(d81.anchor_V, d81.Y_VDC_V):.4f}")
rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=8)
F = ["f_GHz", "pin_avail", "anchor_V", "mech_b", "proto_r15", "proto_r20"]
def fer(d):
    x = pd.DataFrame(index=d.index)
    for c in F[:3]: x[c] = d[c]
    x["mech_b"] = (d.mech == "bent").astype(int)
    x["proto_r15"] = (d.proto == "r15").astype(int); x["proto_r20"] = (d.proto == "r20").astype(int)
    return x
rf.fit(fer(d81), d81.Y_VDC_V)
rep.append(f"RF(+anchor 特征) EX = {mae(rf.predict(fer(d18)), d18.Y_VDC_V):.4f} | ", )

for arm, cols in ARMS.items():
    ntok = 0; ti_all, ex_all = [], []
    for seed in [42, 2024, 7, 123, 99]:
        gkf = GroupKFold(n_splits=5); errs = []
        _, ntok = row_tokens(d81, cols)
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            pr = run(seed, d81.iloc[tr_i], d81.iloc[te_i], cols, ntok)
            errs.append(mae(pr, d81.iloc[te_i].Y_VDC_V.values))
        ti = np.mean(errs)
        pr = run(seed, d81, d18, cols, ntok)
        ex = mae(pr, d18.Y_VDC_V.values)
        ti_all.append(ti); ex_all.append(ex)
    verdict = "WIN" if np.mean(ex_all) < mae(d18.anchor_V, d18.Y_VDC_V) else "LOSE(vs anchor)"
    rep.append(f"{arm}: TI={np.mean(ti_all):.4f}±{np.std(ti_all):.4f}  EX={np.mean(ex_all):.4f}±{np.std(ex_all):.4f}  [{verdict}]")
    rep.append(f"      per-seed EX: {[round(x,4) for x in ex_all]}")

out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/g1_v3_inno_report.md", "w", encoding="utf-8").write(out)
