# -*- coding: utf-8 -*-
"""TEXWPT P1+P2: v4 stronger model + named-baseline army, same protocol, full benchmark.
Protocol: TI = 81785 GroupKFold-by-f 5-fold x 3 seeds; EX = 81785->81813 zero-shot x 3 seeds.
Baselines (named, non-PLM): Ridge/KNN/SVR/RF/XGB/(LightGBM/CatBoost if avail)/GP/TabPFN/(TabNet/rtdl if avail) + PureAnchor.
Ours: v4 = Transformer direct-regression + physics bases + anchor + Smith/Q (full feature), I7 = gated-residual arm.
"""
import os, random, time, json, traceback
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7]
t0 = time.time()
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv"); df["pool"] = df["pool"].astype(str)
an = pd.read_csv(f"{ROOT}/data/processed/anchors.csv")

# ---------- 特征构造（复用 v2/v3 修正版） ----------
g81 = an[an.anchor_id == "81785_Fig4c_gain_air"].sort_values("x1")
ads18v = an[an.anchor_id == "81813_Fig5_ADS_VDC"]; ads81e = an[an.anchor_id == "81785_Fig2_ADS_eff"]
r3 = an[an.anchor_id == "81813_Fig3rect_ADS_VDC"].merge(an[an.anchor_id == "81813_Fig3rect_ADS_eff"], on=["x1"], suffixes=("_v", "_e"))
R_EST = float(np.median((r3.y_val_v ** 2) / (10 ** (-40 / 10) * r3.y_val_e)))
z18g = an[an.anchor_id == "81813_Fig6_HFSS_Z"]; Z18R = z18g[z18g.x2 == 0].sort_values("x1"); Z18I = z18g[z18g.x2 == 1].sort_values("x1")
z81g = an[an.anchor_id == "81785_Fig3air_HFSS_Z"]; Z81R = z81g[z81g.x2 == 0].sort_values("x1"); Z81I = z81g[z81g.x2 == 3].sort_values("x1")

lam = 0.125
PG = 20 * np.log10(lam / (4 * np.pi * df["dist_m"].clip(lower=0.05).fillna(0.5)))
gr81 = np.interp(df["f_GHz"], g81.x1.values, g81.y_val.values)
df["pin_avail"] = np.where(df.pool == "81785", 27.0 + gr81 + PG, df["Pin_dBm"])
df["pin_lin_W"] = 10 ** ((df["pin_avail"] - 30) / 10)

def nn2(ads, R, X):
    d = (ads.x1.values - R) ** 2 + (ads.x2.values - X) ** 2
    return float(ads.y_val.values[np.argmin(d)])

zr18 = np.interp(df["f_GHz"], Z18R.x1.values, Z18R.y_val.values); zi18 = np.interp(df["f_GHz"], Z18I.x1.values, Z18I.y_val.values)
zr81 = np.interp(df["f_GHz"], Z81R.x1.values, Z81R.y_val.values); zi81 = np.interp(df["f_GHz"], Z81I.x1.values, Z81I.y_val.values)
ads_vdc18 = np.array([nn2(ads18v, a, b) for a, b in zip(zr18, zi18)])
ads_eff81 = np.array([nn2(ads81e, a, b) for a, b in zip(zr81, zi81)])
df["pin_off_db"] = df["pin_avail"].values + 10.0
df["anchor_V"] = np.where(df.pool == "81785", np.sqrt(df["pin_lin_W"] * ads_eff81 * R_EST), ads_vdc18 * 10 ** (df["pin_off_db"] / 20))
df.loc[df.anchor_V <= 0.001, "anchor_V"] = 0.001
df["anchor_dB"] = 20 * np.log10(df["anchor_V"])

for pool, ZR, ZI in [("81785", Z81R, Z81I), ("81813", Z18R, Z18I)]:
    m = df.pool == pool
    zc_r = float(np.interp(2.4, ZR.x1.values, ZR.y_val.values)); zc_i = float(np.interp(2.4, ZI.x1.values, ZI.y_val.values))
    zr = np.interp(df.loc[m, "f_GHz"], ZR.x1.values, ZR.y_val.values); zi = np.interp(df.loc[m, "f_GHz"], ZI.x1.values, ZI.y_val.values)
    dz = zr - zc_r + 1j * (zi - zc_i)
    df.loc[m, "smith_ang"] = np.angle(dz); df.loc[m, "smith_norm"] = np.abs(dz) / max(np.abs(dz).max(), 1e-6)
df["smith_ang"] = df["smith_ang"].fillna(0.0); df["smith_norm"] = df["smith_norm"].fillna(0.0)
for (pool, mech), g in df.dropna(subset=["Y_VDC_V"]).groupby(["pool", "mech"]):
    gg = g.groupby("f_GHz")["Y_VDC_V"].mean()
    if len(gg) < 5: continue
    fpk = gg.idxmax(); above = gg[gg >= gg.max() / 2]
    df.loc[(df.pool == pool) & (df.mech == mech), "Q_proxy"] = fpk / max(above.index.max() - above.index.min(), 0.01)
df["Q_proxy"] = df["Q_proxy"].fillna(df["Q_proxy"].median())
df["sq_shape"] = 10 ** (df["pin_off_db"] / 20)
df["sat_shape"] = 1 / (1 + np.exp(-(df["pin_avail"] + 8) / 2))

d81 = df[(df.pool == "81785") & df.Y_VDC_V.notna()].copy()
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()].copy()

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

# ---------- baseline 表格模型（sklearn 口径） ----------
def get_baselines():
    out = {"Ridge": lambda s: Ridge(alpha=1.0), "KNN": lambda s: KNeighborsRegressor(5), "SVR": lambda s: SVR(C=3.0, epsilon=0.01),
           "RF": lambda s: RandomForestRegressor(n_estimators=400, random_state=s, n_jobs=8),
           "GP": lambda s: GaussianProcessRegressor(kernel=ConstantKernel(1.0) * RBF(1.0), normalize_y=True, random_state=s)}
    try:
        from xgboost import XGBRegressor
        out["XGBoost"] = lambda s: XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05, random_state=s, n_jobs=8)
    except Exception: pass
    try:
        from lightgbm import LGBMRegressor
        out["LightGBM"] = lambda s: LGBMRegressor(n_estimators=500, num_leaves=15, learning_rate=0.05, random_state=s, n_jobs=8, verbose=-1)
    except Exception: pass
    try:
        from catboost import CatBoostRegressor
        out["CatBoost"] = lambda s: CatBoostRegressor(iterations=500, depth=5, learning_rate=0.05, random_seed=s, verbose=0)
    except Exception: pass
    return out

AVAIL_NOTE = []
try:
    from tabpfn import TabPFNRegressor
    TABPFN_CK = f"{ROOT}/models/tabpfn_v2reg/tabpfn-v2-regressor.ckpt"
    HAS_TABPFN = True
except Exception:
    HAS_TABPFN = False; AVAIL_NOTE.append("TabPFN import failed")
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
    HAS_TABNET = True
except Exception:
    HAS_TABNET = False; AVAIL_NOTE.append("TabNet(pytorch-tabnet) not installed")

NUMC = ["f_GHz", "pin_avail", "pin_off_db", "df_res_GHz", "anchor_V", "anchor_dB",
        "sq_shape", "sat_shape", "smith_ang", "smith_norm", "Q_proxy"]
def tab_feats(d):
    x = pd.DataFrame(index=d.index)
    for c in NUMC: x[c] = d[c].fillna(0.0)
    x["mech_b"] = (d.mech == "bent").astype(int)
    x["proto_r15"] = (d.proto == "r15").astype(int); x["proto_r20"] = (d.proto == "r20").astype(int)
    x["coup_body"] = (d.coup == "body").astype(int)
    return x

# ---------- 我们的模型 ----------
class TokenNet(nn.Module):
    def __init__(self, ntok, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(ntok, d)
        self.emb = nn.Parameter(torch.randn(ntok, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
def tok_matrix(d, cols):
    FILL = {c: 0.0 for c in cols}; FILL.update({"f_GHz": 2.4, "pin_avail": -10.0, "pin_off_db": 0.0,
        "df_res_GHz": 0.0, "anchor_V": 0.15, "anchor_dB": -16.0, "sq_shape": 1.0, "sat_shape": 0.5,
        "smith_ang": 0.0, "smith_norm": 0.0, "Q_proxy": 0.0})
    xs = [d[c].fillna(FILL[c]).values.astype(np.float32) for c in cols]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)

def run_ours(seed, tr, te, cols, mode="direct"):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    NTOK = len(cols) + 7
    mu_s = {c: (tr[c].fillna(0.0).mean(), max(tr[c].fillna(0.0).std(), 1e-4)) for c in cols}
    def sc(mx):
        o = mx.copy()
        for j, c in enumerate(cols): o[:, j] = (o[:, j] - mu_s[c][0]) / mu_s[c][1]
        return o
    Xtr = torch.tensor(sc(tok_matrix(tr, cols)), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(sc(tok_matrix(te, cols)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NTOK, device=DEV).expand(len(tr), -1); tidt = torch.arange(NTOK, device=DEV).expand(len(te), -1)
    ytr = torch.tensor(tr["Y_VDC_V"].values.astype(np.float32), device=DEV)
    anchor_tr = torch.tensor(tr["anchor_V"].values.astype(np.float32), device=DEV)
    anchor_te = torch.tensor(te["anchor_V"].values.astype(np.float32), device=DEV)
    net = TokenNet(NTOK).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(100):  # MFM denoise
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(cols):] = False
        Xi = Xtr.clone(); Xi[m] = 0.0
        loss = ((net(Xi, tid) - net(torch.where(m, Xtr, Xi), tid)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    grp = (tr["mech"].astype(str) + tr["proto"].astype(str)).astype("category").cat.codes.values.astype(np.int64)
    yt = torch.tensor(grp, device=DEV)
    for ep in range(60):  # SupCon
        net.train(); p = net.pj(net(Xtr, tid))
        mask = (yt[:, None] == yt[None, :]) & (~torch.eye(len(yt), dtype=torch.bool, device=DEV))
        if mask.sum() == 0: break
        sim = (p @ p.T / 0.07).masked_fill(torch.eye(len(yt), dtype=torch.bool, device=DEV), -1e9)
        loss = nn.functional.cross_entropy(sim, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    for ep in range(350):
        net.train(); h = net(Xtr, tid)
        pred = net.head(h).squeeze(-1)
        if mode == "gated":  # I7: yhat = anchor + g * residual
            pred = anchor_tr + net.gate(h).squeeze(-1) * pred
        loss = ((pred - ytr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h = net(Xte, tidt); pred = net.head(h).squeeze(-1)
        if mode == "gated":
            pred = anchor_te + net.gate(h).squeeze(-1) * pred
    return np.clip(pred.cpu().numpy(), 0.01, 2.0)

FULL = NUMC  # v4 全特征
results = {}
def bench(name, fn):
    ti_all, ex_all = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            errs.append(fn(d81.iloc[tr_i], d81.iloc[te_i], seed))
        ti_all.append(np.mean(errs))
        ex_all.append(fn(d81, d18, seed))
    results[name] = (float(np.mean(ti_all)), float(np.std(ti_all)), float(np.mean(ex_all)), float(np.std(ex_all)))
    r = results[name]
    print(f"[{name}] TI={r[0]:.4f}±{r[1]:.4f}  EX={r[2]:.4f}±{r[3]:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- baselines (sklearn 口径) ----
def sk_fn(maker):
    def f(tr, te, seed):
        m = maker(seed)
        sc = StandardScaler().fit(tab_feats(tr))
        m.fit(sc.transform(tab_feats(tr)), tr.Y_VDC_V.values)
        return mae(m.predict(sc.transform(tab_feats(te))), te.Y_VDC_V.values)
    return f
for name, maker in get_baselines().items():
    bench(name, sk_fn(maker))

# PureAnchor
bench("PureAnchor(sim)", lambda tr, te, s: mae(te.anchor_V.values, te.Y_VDC_V.values))

# TabPFN
if HAS_TABPFN:
    def tabpfn_fn(tr, te, seed):
        m = TabPFNRegressor(model_path=TABPFN_CK, device="cpu")
        m.fit(tab_feats(tr).values, tr.Y_VDC_V.values)
        return mae(m.predict(tab_feats(te).values), te.Y_VDC_V.values)
    bench("TabPFN-v2", tabpfn_fn)

# TabNet
if HAS_TABNET:
    def tabnet_fn(tr, te, seed):
        torch.manual_seed(seed)
        m = TabNetRegressor(seed=seed, verbose=0, device_name="cpu")
        m.fit(tab_feats(tr).values, tr.Y_VDC_V.values.reshape(-1, 1), max_epochs=200, patience=30, batch_size=64)
        return mae(m.predict(tab_feats(te).values).ravel(), te.Y_VDC_V.values)
    bench("TabNet", tabnet_fn)

# ---- ours ----
bench("TEXWPT-v4(full, direct)", lambda tr, te, s: mae(run_ours(s, tr, te, FULL, "direct"), te.Y_VDC_V.values))
bench("TEXWPT-I7(gated-res)", lambda tr, te, s: mae(run_ours(s, tr, te, FULL, "gated"), te.Y_VDC_V.values))
bench("TEXWPT-v4-lite(no physics)", lambda tr, te, s: mae(run_ours(s, tr, te, ["f_GHz", "pin_avail", "df_res_GHz", "anchor_V"], "direct"), te.Y_VDC_V.values))

# ---- 汇总表 ----
lines = [f"# P1/P2 full benchmark (2026-09-27, {time.time()-t0:.0f}s)", "",
         "| 模型 | TI MAE_V | EX MAE_V |", "|---|---|---|"]
for k, (tm, ts, em, es) in sorted(results.items(), key=lambda kv: kv[1][2]):
    lines.append(f"| {k} | {tm:.4f}±{ts:.4f} | {em:.4f}±{es:.4f} |")
lines.append("\n渠道缺失记录: " + "; ".join(AVAIL_NOTE) if AVAIL_NOTE else "")
out = "\n".join(lines)
open(f"{ROOT}/data/qa/p1_full_benchmark.md", "w", encoding="utf-8").write(out)
json.dump(results, open(f"{ROOT}/data/qa/p1_results.json", "w"), indent=1)
print(out)
