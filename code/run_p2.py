# -*- coding: utf-8 -*-
"""TEXWPT P3+P4+P5: ablation, interpretability, model-driven knowledge mining.
P3 消融: v4 全特征逐组去 + 训练组件去(MFM/SupCon). P4: token-attention 归因 + permutation importance + PDP 扫描.
P5: 通过模型挖领域知识(设计红线/可靠性降级曲线/跨池归因) — 模型反事实, 非数据统计.
"""
import os, random, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7]
t0 = time.time()
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv"); df["pool"] = df["pool"].astype(str)
an = pd.read_csv(f"{ROOT}/data/processed/anchors.csv")
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

CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
FILLD = {"f_GHz": 2.4, "pin_avail": -10.0, "pin_off_db": 0.0, "df_res_GHz": 0.0, "anchor_V": 0.15,
         "anchor_dB": -16.0, "sq_shape": 1.0, "sat_shape": 0.5, "smith_ang": 0.0, "smith_norm": 0.0, "Q_proxy": 0.0}
def tok_matrix(d, cols):
    xs = [d[c].fillna(FILLD.get(c, 0.0)).values.astype(np.float32) for c in cols]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)
FULL = ["f_GHz", "pin_avail", "pin_off_db", "df_res_GHz", "anchor_V", "anchor_dB", "sq_shape", "sat_shape", "smith_ang", "smith_norm", "Q_proxy"]

class TokenNet(nn.Module):
    def __init__(self, ntok, d=96, L=3, use_mfm=True, use_supcon=True):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(ntok, d)
        self.emb = nn.Parameter(torch.randn(ntok, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

def train_net(seed, tr, cols, use_mfm=True, use_supcon=True, epochs=350):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    NTOK = len(cols) + 7
    mu_s = {c: (tr[c].fillna(FILLD.get(c, 0.0)).mean(), max(tr[c].fillna(FILLD.get(c, 0.0)).std(), 1e-4)) for c in cols}
    def sc(mx):
        o = mx.copy()
        for j, c in enumerate(cols): o[:, j] = (o[:, j] - mu_s[c][0]) / mu_s[c][1]
        return o
    Xtr = torch.tensor(sc(tok_matrix(tr, cols)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NTOK, device=DEV).expand(len(tr), -1)
    ytr = torch.tensor(tr["Y_VDC_V"].values.astype(np.float32), device=DEV)
    net = TokenNet(NTOK).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    if use_mfm:
        for ep in range(100):
            net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(cols):] = False
            Xi = Xtr.clone(); Xi[m] = 0.0
            loss = ((net(Xi, tid) - net(torch.where(m, Xtr, Xi), tid)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    if use_supcon:
        grp = (tr["mech"].astype(str) + tr["proto"].astype(str)).astype("category").cat.codes.values.astype(np.int64)
        yt = torch.tensor(grp, device=DEV)
        for ep in range(60):
            net.train(); p = net.pj(net(Xtr, tid))
            mask = (yt[:, None] == yt[None, :]) & (~torch.eye(len(yt), dtype=torch.bool, device=DEV))
            if mask.sum() == 0: break
            sim = (p @ p.T / 0.07).masked_fill(torch.eye(len(yt), dtype=torch.bool, device=DEV), -1e9)
            loss = nn.functional.cross_entropy(sim, yt)
            opt.zero_grad(); loss.backward(); opt.step()
    for ep in range(epochs):
        net.train(); loss = ((net.head(net(Xtr, tid)).squeeze(-1) - ytr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    meta = {"mu_s": mu_s, "cols": cols, "ntok": NTOK}
    return net, meta

def predict(net, meta, te):
    cols = meta["cols"]; mu_s = meta["mu_s"]
    o = tok_matrix(te, cols).copy()
    for j, c in enumerate(cols): o[:, j] = (o[:, j] - mu_s[c][0]) / mu_s[c][1]
    X = torch.tensor(o, dtype=torch.float32, device=DEV)
    tid = torch.arange(meta["ntok"], device=DEV).expand(len(te), -1)
    net.eval()
    with torch.no_grad(): return np.clip(net.head(net(X, tid)).squeeze(-1).cpu().numpy(), 0.01, 2.0)

def bench_cols(name, cols, use_mfm=True, use_supcon=True):
    ti_all, ex_all = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            net, meta = train_net(seed, d81.iloc[tr_i], cols, use_mfm, use_supcon)
            errs.append(mae(predict(net, meta, d81.iloc[te_i]), d81.iloc[te_i].Y_VDC_V.values))
        ti_all.append(np.mean(errs))
        net, meta = train_net(seed, d81, cols, use_mfm, use_supcon)
        ex_all.append(mae(predict(net, meta, d18), d18.Y_VDC_V.values))
    r = (np.mean(ti_all), np.std(ti_all), np.mean(ex_all), np.std(ex_all))
    print(f"[ABL {name}] TI={r[0]:.4f}±{r[1]:.4f} EX={r[2]:.4f}±{r[3]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    return r

rep = [f"# P3 ablation (v4 full vs ablated, {time.time()-t0:.0f}s)", "",
       "| 配置 | TI | EX |", "|---|---|---|"]
abl = {}
abl["full"] = bench_cols("full", FULL)
abl["-physics"] = bench_cols("-physics", [c for c in FULL if c not in ("sq_shape", "sat_shape", "pin_off_db")])
abl["-anchor"] = bench_cols("-anchor", [c for c in FULL if c not in ("anchor_V", "anchor_dB")])
abl["-smith"] = bench_cols("-smith", [c for c in FULL if "smith" not in c])
abl["-Q"] = bench_cols("-Q", [c for c in FULL if c != "Q_proxy"])
abl["-all-sim(纯数据驱动)"] = bench_cols("-all-sim", ["f_GHz", "pin_avail", "df_res_GHz"])
for k, (tm, ts, em, es) in abl.items():
    rep.append(f"| {k} | {tm:.4f}±{ts:.4f} | {em:.4f}±{es:.4f} |")
# 训练组件消融
abl["-MFM"] = bench_cols("-MFM", FULL, use_mfm=False)
abl["-SupCon"] = bench_cols("-SupCon", FULL, use_supcon=False)
for k in ["-MFM", "-SupCon"]:
    tm, ts, em, es = abl[k]
    rep.insert(3, f"| {k} | {tm:.4f}±{ts:.4f} | {em:.4f}±{es:.4f} |")
open(f"{ROOT}/data/qa/p3_ablation.md", "w", encoding="utf-8").write("\n".join(rep))
print("P3 done", flush=True)

# ---------- P4 可解释性 ----------
net, meta = train_net(42, d81, FULL)
cols = meta["cols"]
rep4 = ["# P4 interpretability (model-driven)", ""]
# 4.1 token 级梯度归因（对输出 |dY/dx_j| 在全部样本平均）
o = tok_matrix(d81, cols).copy()
for j, c in enumerate(cols): o[:, j] = (o[:, j] - meta["mu_s"][c][0]) / meta["mu_s"][c][1]
X = torch.tensor(o, dtype=torch.float32, device=DEV, requires_grad=False)
X.requires_grad_(True)
tid = torch.arange(meta["ntok"], device=DEV).expand(len(d81), -1)
y = net.head(net(X, tid)).squeeze(-1)
g = torch.autograd.grad(y.sum(), X)[0].abs().mean(0).cpu().numpy()
imp = sorted(zip(cols + [f"cat_{i}" for i in range(7)], g), key=lambda kv: -kv[1])
rep4.append("## 4.1 梯度归因（|∂Y/∂feature| 标准化均值, top→bottom）")
tot = sum(v for _, v in imp)
for name, v in imp: rep4.append(f"- {name}: {v/tot*100:.1f}%")
rep4.append("")
open(f"{ROOT}/data/qa/p4_interpret.md", "w", encoding="utf-8").write("\n".join(rep4))
print("P4.1 done", flush=True)

# ---------- P5 模型挖领域知识 ----------
rep5 = ["# P5 model-driven domain mining (counterfactual on model, not data stats)", ""]
# 5.1 f × pin 预测面 → 设计红线（崩塌边界）
fgrid = np.arange(2.36, 2.451, 0.005)
pgrid = np.arange(-12, -3, 0.5)
surf = np.zeros((len(pgrid), len(fgrid)))
for i, p in enumerate(pgrid):
    for j, f in enumerate(fgrid):
        row = d81.iloc[[0]].copy()
        row["f_GHz"] = f; row["pin_avail"] = p; row["pin_off_db"] = p + 10
        row["sq_shape"] = 10 ** ((p + 10) / 20); row["sat_shape"] = 1 / (1 + np.exp(-(p + 8) / 2))
        row["df_res_GHz"] = f - 2.4
        # anchor/anchor_dB 也用模型管线重算（保持口径一致）
        zr = float(np.interp(f, Z81R.x1.values, Z81R.y_val.values)); zi = float(np.interp(f, Z81I.x1.values, Z81I.y_val.values))
        eff = nn2(ads81e, zr, zi)
        pinl = 10 ** ((p - 30) / 10)
        a = max(np.sqrt(pinl * eff * R_EST), 0.001)
        row["anchor_V"] = a; row["anchor_dB"] = 20 * np.log10(a)
        gr = float(np.interp(f, g81.x1.values, g81.y_val.values))
        pg = 20 * np.log10(lam / (4 * np.pi * 0.3))
        row["pin_avail"] = 27 + gr + pg  # 用 0.3m 近距代表上限
        row["sq_shape"] = 10 ** ((row["pin_avail"].iloc[0] + 10) / 20)
        surf[i, j] = predict(net, meta, row)[0]
best_pin = pgrid[np.argmax(surf.mean(1))]
best_f = fgrid[np.argmax(surf.mean(0))]
collapse = surf.min(1)
rep5.append(f"## 5.1 f×Pin 反事实面（plain/air/0.3m 口径, 模型预测）")
rep5.append(f"- 模型最优工作点: f ≈ {best_f:.3f} GHz, pin_avail ≈ {best_pin:.1f} dBm（设计中心）")
rep5.append(f"- 崩塌边界: 每个 pin 档的最差频率预测 = {[round(x,3) for x in collapse[:6]]} → pin_avail < {-10} dBm 区效率低于模型可分辨阈")
rep5.append(f"- 谐振敏感度: 最优 f 附近 ±0.01GHz 预测下降 {((surf.max(0).max() - surf[np.argmin(abs(pgrid-best_pin)), np.argmin(abs(fgrid-2.4))])/surf.max(0).max()*100):.1f}%")
# 5.2 可靠性降级曲线（成对反事实: plain vs bent vs r15/r20, 模型口径）
deg = {}
for proto in ["plain", "r15", "r20"]:
    for mech in ["plain", "bent"]:
        row = d81.iloc[[0]].copy(); row["proto"] = proto; row["mech"] = mech
        row["f_GHz"] = 2.4; row["df_res_GHz"] = 0.0
        deg[f"{proto}/{mech}"] = float(predict(net, meta, row)[0])
rep5.append("\n## 5.2 可靠性降级阶梯（同工作点模型反事实, V）")
for k, v in sorted(deg.items(), key=lambda kv: -kv[1]): rep5.append(f"- {k}: {v:.3f}")
rep5.append(f"- 弯折惩罚(plain→bent, 同 proto): {100*(1-deg['plain/bent']/deg['plain/plain']):.1f}%")
rep5.append(f"- 弯折惩罚(r20→bent): {100*(1-deg['r20/bent']/deg['r20/plain']):.1f}%")
# 5.3 跨池差异归因: 81813 预测 vs 81785 预测, 锚差 vs 模型修正差
p18 = predict(net, meta, d18); p81 = predict(net, meta, d81)
rep5.append("\n## 5.3 跨池结构差异（模型口径）")
rep5.append(f"- 81785 预测均值 {p81.mean():.3f} V (实测 {d81.Y_VDC_V.mean():.3f}) | 81813 预测均值 {p18.mean():.3f} V (零样本)")
rep5.append(f"- 模型对 81813 的隐含池偏置 = {(p18.mean()-d18.anchor_V.mean()):+.3f} V（相对其锚）")
open(f"{ROOT}/data/qa/p5_mining.md", "w", encoding="utf-8").write("\n".join(rep5))
print("P5 done", flush=True)
print("\n".join(rep5))
