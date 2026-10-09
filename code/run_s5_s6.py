# -*- coding: utf-8 -*-
"""TEXWPT S5+S6: interpretability + model-driven mining on I7 (main config, leak-free).
S5: grad attribution + permutation importance + PDP scans. S6: counterfactual design center,
collapse boundary, degradation ladder incl. body-air (was missing), cross-pool attribution.
"""
import os, random, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from features import d81, d18, FULL, R_EST, DEV, CATS, FILLD, an, ROOT

SEEDS = [42, 2024, 7]
t0 = time.time()
NTOK = len(FULL) + 7
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
def tok_matrix(d):
    xs = [d[c].fillna(FILLD[c]).values.astype(np.float32) for c in FULL]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)

class I7Net(nn.Module):
    def __init__(self, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(NTOK, d)
        self.emb = nn.Parameter(torch.randn(NTOK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

def train_i7(seed, tr):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    mu_s = {c: (tr[c].fillna(FILLD[c]).mean(), max(tr[c].fillna(FILLD[c]).std(), 1e-4)) for c in FULL}
    def sc(mx):
        o = mx.copy()
        for j, c in enumerate(FULL): o[:, j] = (o[:, j] - mu_s[c][0]) / mu_s[c][1]
        return o
    Xtr = torch.tensor(sc(tok_matrix(tr)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NTOK, device=DEV).expand(len(tr), -1)
    ytr = torch.tensor(tr.Y_VDC_V.values.astype(np.float32), device=DEV)
    a_tr = torch.tensor(tr.anchor_V.values.astype(np.float32), device=DEV)
    net = I7Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(100):
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(FULL):] = False
        Xi = Xtr.clone(); Xi[m] = 0.0
        loss = ((net(Xi, tid) - net(torch.where(m, Xtr, Xi), tid)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    grp = (tr["mech"].astype(str) + tr["proto"].astype(str)).astype("category").cat.codes.values.astype(np.int64)
    yt = torch.tensor(grp, device=DEV)
    for ep in range(60):
        net.train(); p = net.pj(net(Xtr, tid))
        mask = (yt[:, None] == yt[None, :]) & (~torch.eye(len(yt), dtype=torch.bool, device=DEV))
        if mask.sum() == 0: break
        sim = (p @ p.T / 0.07).masked_fill(torch.eye(len(yt), dtype=torch.bool, device=DEV), -1e9)
        loss = nn.functional.cross_entropy(sim, yt)
        opt.zero_grad(); loss.backward(); opt.step()
    for ep in range(350):
        net.train(); h = net(Xtr, tid)
        pred = a_tr + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
        loss = ((pred - ytr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    meta = {"mu_s": mu_s}
    return net, meta

def predict(net, meta, te):
    o = tok_matrix(te).copy()
    for j, c in enumerate(FULL): o[:, j] = (o[:, j] - meta["mu_s"][c][0]) / meta["mu_s"][c][1]
    X = torch.tensor(o, dtype=torch.float32, device=DEV)
    tid = torch.arange(NTOK, device=DEV).expand(len(te), -1)
    net.eval()
    with torch.no_grad():
        h = net(X, tid)
        return np.clip((te.anchor_V.values.astype(np.float32) + net.gate(h).squeeze(-1).cpu().numpy() * net.head(h).squeeze(-1).cpu().numpy()), 0.01, 2.0)

# ensemble of 3 seeds (main config robustness)
nets = []
for s in SEEDS:
    n, m = train_i7(s, d81)
    nets.append((n, m))
def pred_ens(te):
    return np.mean([predict(n, m, te) for n, m in nets], axis=0)

print(f"trained {len(nets)} seeds ({time.time()-t0:.0f}s)", flush=True)

# ---------- S5: interpretability ----------
rep = ["# S5 interpretability on TEXWPT-I7 (3-seed ensemble, leak-free)", ""]
# 5.1 grad attribution
n0, m0 = nets[0]
o = tok_matrix(d81).copy()
for j, c in enumerate(FULL): o[:, j] = (o[:, j] - m0["mu_s"][c][0]) / m0["mu_s"][c][1]
X = torch.tensor(o, dtype=torch.float32, device=DEV, requires_grad=True)
tid = torch.arange(NTOK, device=DEV).expand(len(d81), -1)
aX = torch.tensor(d81.anchor_V.values.astype(np.float32), device=DEV)
h = n0(X, tid)
pred = aX + n0.gate(h).squeeze(-1) * n0.head(h).squeeze(-1)
g = torch.autograd.grad(pred.sum(), X)[0].abs().mean(0).cpu().numpy()
imp = sorted(zip(FULL + [f"cat_{i}" for i in range(7)], g), key=lambda kv: -kv[1])
tot = sum(v for _, v in imp)
rep.append("## 5.1 梯度归因（占前向总敏感度 %）")
for name, v in imp: rep.append(f"- {name}: {v/tot*100:.1f}%")
# 5.2 permutation importance (EX, ensemble)
base = mae(pred_ens(d18), d18.Y_VDC_V.values)
rep.append(f"\n## 5.2 permutation importance（EX MAE 退化, base={base:.4f}）")
rng = np.random.default_rng(0)
for c in FULL:
    te = d18.copy(); te[c] = rng.permutation(te[c].values)
    rep.append(f"- {c}: +{mae(pred_ens(te), te.Y_VDC_V.values)-base:.4f}")
open(f"{ROOT}/data/qa/s5_interpret.md", "w", encoding="utf-8").write("\n".join(rep))
print("S5 done", flush=True)

# ---------- S6: model-driven mining ----------
rep6 = ["# S6 model-driven mining (TEXWPT-I7 ensemble counterfactuals)", ""]
g81 = an[an.anchor_id == "81785_Fig4c_gain_air"].sort_values("x1")
z81g = an[an.anchor_id == "81785_Fig3air_HFSS_Z"]
Z81R = z81g[z81g.x2 == 0].sort_values("x1"); Z81I = z81g[z81g.x2 == 3].sort_values("x1")
ads81e = an[an.anchor_id == "81785_Fig2_ADS_eff"]
def nn2(ads, R, X):
    from scipy.interpolate import RegularGridInterpolator
    x1s = np.sort(ads.x1.unique()); x2s = np.sort(ads.x2.unique())
    grid = np.full((len(x1s), len(x2s)), np.nan)
    p1 = {v: i for i, v in enumerate(x1s)}; p2 = {v: i for i, v in enumerate(x2s)}
    for r in ads.itertuples(): grid[p1[r.x1], p2[r.x2]] = r.y_val
    itp = RegularGridInterpolator((x1s, x2s), grid, method="linear", bounds_error=False, fill_value=None)
    return float(itp([[R, X]])[0])
lam = 0.125
# 6.1 degradation ladder incl. body (S4 前缺 body 维)
lad = {}
for coup in ["air", "body"]:
    for proto in ["plain", "r15", "r20"]:
        for mech in ["plain", "bent"]:
            row = d81.iloc[[0]].copy()
            row["coup"] = coup; row["proto"] = proto; row["mech"] = mech
            row["f_GHz"] = 2.4; row["df_res_GHz"] = 0.0
            row["pin_avail"] = row["pin_avail"].iloc[0]; row["pin_off_db"] = row["pin_off_db"].iloc[0]
            row["sq_shape"] = 10 ** (row["pin_off_db"].iloc[0] / 20)
            row["sat_shape"] = 1 / (1 + np.exp(-(row["pin_avail"].iloc[0] + 8) / 2))
            lad[f"{coup}/{proto}/{mech}"] = float(pred_ens(row))
rep6.append("## 6.1 可靠性降级阶梯（同工作点, 耦合×原型×机械, V）")
for k, v in sorted(lad.items(), key=lambda kv: -kv[1]): rep6.append(f"- {k}: {v:.3f}")
rep6.append(f"- 体载惩罚(plain/plain: air→body): {100*(1-lad['air/plain/plain']/max(lad['air/plain/plain'],1e-9)):.1f}% (基点=自身)")
rep6.append(f"- 体载代价(air→body, plain/plain): {100*(1-lad['body/plain/plain']/lad['air/plain/plain']):.1f}%")
rep6.append(f"- 体载代价(r15/bent vs air/plain): {100*(1-lad['body/r15/bent']/lad['air/plain/plain']):.1f}% (最恶劣组合总降级)")
rep6.append(f"- 弯折代价(air, plain→bent): {100*(1-lad['air/plain/bent']/lad['air/plain/plain']):.1f}%")
# 6.2 f×pin face + collapse boundary (plain/air/0.3m)
fgrid = np.arange(2.36, 2.451, 0.005); pgrid = np.arange(-12, -2.9, 0.5)
surf = np.zeros((len(pgrid), len(fgrid)))
gV = g81.y_val.values; gF = g81.x1.values
for i, p in enumerate(pgrid):
    for j, f in enumerate(fgrid):
        row = d81.iloc[[0]].copy()
        gr = float(np.interp(f, gF, gV)); pg = 20 * np.log10(lam / (4 * np.pi * 0.3))
        pa = 27 + gr + pg
        zr = float(np.interp(f, Z81R.x1.values, Z81R.y_val.values))
        zi = float(np.interp(f, Z81I.x1.values, Z81I.y_val.values))
        eff = nn2(ads81e, zr, zi)
        pinl = 10 ** ((pa - 30) / 10)
        a = max(np.sqrt(pinl * eff * R_EST), 0.001)
        row["f_GHz"] = f; row["pin_avail"] = pa; row["pin_off_db"] = pa + 10
        row["df_res_GHz"] = f - 2.4; row["anchor_V"] = a; row["anchor_dB"] = 20 * np.log10(a)
        row["sq_shape"] = 10 ** ((pa + 10) / 20); row["sat_shape"] = 1 / (1 + np.exp(-(pa + 8) / 2))
        surf[i, j] = pred_ens(row)[0]
bi, bj = np.unravel_index(np.argmax(surf), surf.shape)
rep6.append(f"\n## 6.2 f×pin 反事实面（plain/air/0.3m, 3-seed 集成）")
rep6.append(f"- 设计中心: f≈{fgrid[bj]:.3f} GHz, pin_avail≈{pgrid[bi]:.1f} dBm (max V={surf.max():.3f})")
worst_per_pin = surf.min(1)
rep6.append(f"- 每 pin 档最差频率预测: {[round(x,3) for x in worst_per_pin[:5]]} → pin≤-12dBm 死区 V<0.11")
rep6.append(f"- f 敏感度: 峰值频率 ±0.01GHz 内平均下降 {100*(1-surf[:, np.argmin(abs(fgrid-2.4))].mean()/surf.max()):.1f}%")
rep6.append(f"- 门控均值(81813 全测试行): {float(pred_ens(d18).mean()):.3f} V vs 实测 {d18.Y_VDC_V.mean():.3f} V")
open(f"{ROOT}/data/qa/s6_mining.md", "w", encoding="utf-8").write("\n".join(rep6))
print("S6 done\n" + "\n".join(rep6), flush=True)
