# -*- coding: utf-8 -*-
"""TEXWPT P4-INNO2: I8 物理一致性多任务 + I9 R_eff 可学习锚(修系统偏差), 全部 × I7 门控.
臂: A=I7(基准) / B=I7+I9 / C=I7+I8 / D=I7+I9+I8(终配候选). 5 seeds x TI/EX. H58 快筛.
物理: anchor_dB = 10log10(Pin·eff·R) → R_eff 可学习 = +10·log10(R_l/R_EST) dB 偏置 (I9).
      eff_pred ≈ VDC_pred²/(R·Pin_lin) 闭环软约束 (I8), eff 有实测监督.
"""
import os, random, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7, 123, 99]
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
df["sq_shape"] = 10 ** (df["pin_off_db"] / 20)
df["sat_shape"] = 1 / (1 + np.exp(-(df["pin_avail"] + 8) / 2))
d81 = df[(df.pool == "81785") & df.Y_VDC_V.notna()].copy()
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()].copy()
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

NUMC = ["f_GHz", "pin_avail", "pin_off_db", "df_res_GHz", "anchor_dB", "sq_shape", "sat_shape", "smith_ang", "smith_norm"]
CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
FILLD = {c: 0.0 for c in NUMC}; FILLD.update({"f_GHz": 2.4, "pin_avail": -10.0})
def tok_matrix(d, cols):
    xs = [d[c].fillna(FILLD.get(c, 0.0)).values.astype(np.float32) for c in cols]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)
NTOK = len(NUMC) + 7

class I7Net(nn.Module):
    def __init__(self, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(NTOK, d)
        self.emb = nn.Parameter(torch.randn(NTOK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))          # Δ (residual on anchor_dB)
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
        # I9: 可学习 R_eff (物理参数化锚修正), init=R_EST
        self.logR = nn.Parameter(torch.tensor(np.log10(R_EST)))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

def run_arm(seed, tr, te, use_i9, use_i8):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    mu_s = {c: (tr[c].fillna(FILLD.get(c, 0.0)).mean(), max(tr[c].fillna(FILLD.get(c, 0.0)).std(), 1e-4)) for c in NUMC}
    def sc(mx):
        o = mx.copy()
        for j, c in enumerate(NUMC): o[:, j] = (o[:, j] - mu_s[c][0]) / mu_s[c][1]
        return o
    Xtr = torch.tensor(sc(tok_matrix(tr, NUMC)), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(sc(tok_matrix(te, NUMC)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NTOK, device=DEV).expand(len(tr), -1); tidt = torch.arange(NTOK, device=DEV).expand(len(te), -1)
    # 目标: anchor_dB(可含 I9 修正) + Δ
    if use_i9:
        anchor_dB_tr = tr.anchor_dB.values + 10.0 * (tr.logR_l - np.log10(R_EST)) if False else None
    # I9 的 R 可学习在 net 内: anchor_dB_adj = anchor_dB + 10*(logR - log10(R_EST)) 对 81785 行
    is81785_tr = torch.tensor((tr.pool == "81785").values.astype(np.float32), device=DEV)
    is81785_te = torch.tensor((te.pool == "81785").values.astype(np.float32), device=DEV)
    net = I7Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    aDBtr = torch.tensor(tr.anchor_dB.values.astype(np.float32), device=DEV)
    aDBte = torch.tensor(te.anchor_dB.values.astype(np.float32), device=DEV)
    def anchor_db_row(dB, is81):
        return dB + is81 * 10.0 * (net.logR - np.log10(R_EST))  # I9 修正只作用于训练池口径行(图内重算)
    ytr = torch.tensor(tr.Y_VDC_V.values.astype(np.float32), device=DEV)
    pin_tr = torch.tensor(tr.pin_lin_W.fillna(1e-4).values.astype(np.float32), device=DEV)
    # eff 监督 (I8)
    has_eff = torch.tensor(tr.Y_eff_01.notna().values.astype(np.float32), device=DEV)
    yeff = torch.tensor(tr.Y_eff_01.fillna(0.0).values.astype(np.float32), device=DEV)
    for ep in range(100):  # MFM
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(NUMC):] = False
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
        a_tr = 10 ** (anchor_db_row(aDBtr, is81785_tr) / 20) if use_i9 else 10 ** (aDBtr / 20)
        pred = a_tr + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
        loss = ((pred - ytr) ** 2).mean()
        if use_i8:  # 物理闭环: eff_pred = VDC²/(R_eff·Pin) vs 实测 eff
            Rl = 10 ** net.logR
            eff_pred = (pred ** 2) / (Rl * pin_tr.clamp(min=1e-6))
            loss = loss + 0.3 * (((eff_pred.clamp(0, 1) - yeff) ** 2) * has_eff).sum() / has_eff.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h = net(Xte, tidt)
        a_te = 10 ** (anchor_db_row(aDBte, is81785_te) / 20) if use_i9 else 10 ** (aDBte / 20)
        pred = a_te + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
    return np.clip(pred.cpu().numpy(), 0.01, 2.0)

ARMS = {"A_I7": (False, False), "B_I7+I9(R_eff学习)": (True, False), "C_I7+I8(物理闭环)": (False, True), "D_I7+I9+I8(全组合)": (True, True)}
rep = [f"# P4-INNO2 screen (I8 物理闭环 / I9 R_eff 可学习锚, R_EST={R_EST/1000:.2f}k, dev={DEV})", ""]
res = {}
for arm, (u9, u8) in ARMS.items():
    ti_all, ex_all = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            errs.append(mae(run_arm(seed, d81.iloc[tr_i], d81.iloc[te_i], u9, u8), d81.iloc[te_i].Y_VDC_V.values))
        ti_all.append(np.mean(errs))
        ex_all.append(mae(run_arm(seed, d81, d18, u9, u8), d18.Y_VDC_V.values))
    r = (float(np.mean(ti_all)), float(np.std(ti_all)), float(np.mean(ex_all)), float(np.std(ex_all)))
    res[arm] = r
    print(f"[{arm}] TI={r[0]:.4f}±{r[1]:.4f}  EX={r[2]:.4f}±{r[3]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
rep.append("| 臂 | TI | EX | I7基准(0.0357/0.1173) |", )
for arm, (tm, ts, em, es) in res.items():
    verdict = "WIN" if em < 0.1173 else ("TIE" if em < 0.1242 else "LOSE")
    rep.append(f"| {arm} | {tm:.4f}±{ts:.4f} | {em:.4f}±{es:.4f} | EX {verdict} |")
out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/p4_inno2_report.md", "w", encoding="utf-8").write(out)
json.dump(res, open(f"{ROOT}/data/qa/p4_inno2.json", "w"), indent=1)
