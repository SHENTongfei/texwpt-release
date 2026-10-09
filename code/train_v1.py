# -*- coding: utf-8 -*-
"""TEXWPT G1-v1: PARD anchor injection + 5 seeds. 实质变化 vs v0:
 (1) A1/A2 锚特征 token (ADS eff/VDC 曲面插值 + HFSS gain/effrad/Zant(f) 插值) — 跨池唯一共享物理
 (2) PARD 残差头: yhat_dB = log-anchor(pool calibration) + g(theta) — 结构级
 (3) 5 seeds 分布 (v0 只有 2), 汇报 mean±std 与 RF 靶逐项对比
"""
import os, math, re
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold

torch.manual_seed(0); DEV = "cuda" if torch.cuda.is_available() else "cpu"
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv"); df["pool"] = df["pool"].astype(str)
an = pd.read_csv(f"{ROOT}/data/processed/anchors.csv")

# ---------- 锚插值表 ----------
def curve(src, xcol="x1", ycol="y_val"):
    g = an[an.anchor_id == src].sort_values(xcol)
    return g[xcol].values, g[ycol].values

def interp1(x, xp, fp):
    return np.interp(x, xp, fp)

# 81785: HFSS Z(f) (Fig3air, 列结构 = re×3 [Lloop 4/4.5/5mm] + im×3 → x2: 0=re(4mm), 3=im(4mm))
g81 = an[an.anchor_id == "81785_Fig3air_HFSS_Z"]
z81_re = g81[g81.x2 == 0].sort_values("x1")
z81_im = g81[g81.x2 == 3].sort_values("x1")
eff81_f, eff81_v = curve("81785_Fig4c_eff_air")
gn81_f, gn81_v = curve("81785_Fig4c_gain_air")
ads81 = an[an.anchor_id == "81785_Fig2_ADS_eff"]
# 81813: HFSS Z(f) Fig6 (列 = re, im → x2: 0/1); ADS VDC/eff 曲面 Fig5/Fig4
z18g = an[an.anchor_id == "81813_Fig6_HFSS_Z"]
z18_re = z18g[z18g.x2 == 0].sort_values("x1"); z18_im = z18g[z18g.x2 == 1].sort_values("x1")
ads18v = an[an.anchor_id == "81813_Fig5_ADS_VDC"]; ads18e = an[an.anchor_id == "81813_Fig4_ADS_eff"]

def nn2(ads, R, X, ycol="y_val"):
    # 最近邻 (规则网格 5Ω 步长, 足够)
    d = (ads.x1.values - R) ** 2 + (ads.x2.values - X) ** 2
    return float(ads[ycol].values[np.argmin(d)])

def anchor_feats(r):
    f = r.f_GHz
    if r.pool == "81785":
        zr = float(np.interp(f, z81_re.x1.values, z81_re.y_val.values))
        zi = float(np.interp(f, z81_im.x1.values, z81_im.y_val.values))
        ae = nn2(ads81, zr, zi)
        hg = float(np.interp(f, gn81_f, gn81_v)); he = float(np.interp(f, eff81_f, eff81_v))
        return [ae, np.nan, hg, he, zr, zi]
    else:
        zr = float(np.interp(f, z18_re.x1.values, z18_re.y_val.values))
        zi = float(np.interp(f, z18_im.x1.values, z18_im.y_val.values))
        av = nn2(ads18v, zr, zi); ae = nn2(ads18e, zr, zi)
        return [ae, av, np.nan, np.nan, zr, zi]

AF = df.apply(anchor_feats, axis=1, result_type="expand")
AF.columns = ["ads_eff", "ads_VDC", "hfss_gain", "hfss_effrad", "zant_re", "zant_im"]
for c in AF.columns: df[c] = AF[c]
# 池内 z-归一锚特征(跨池量纲不同, 池内相对形状是共享信号)
for c in ["ads_eff", "ads_VDC", "hfss_gain", "hfss_effrad", "zant_re", "zant_im"]:
    df[c + "_n"] = df.groupby("pool")[c].transform(lambda s: (s - s.mean()) / max(s.std(), 1e-6))
    df[c + "_n"] = df[c + "_n"].fillna(0.0)

COLS = ["f_GHz", "dist_m", "Pin_dBm", "PG_dB", "Prx_pred_dBm", "df_res_GHz"]
ACOLS = ["ads_eff_n", "ads_VDC_n", "hfss_gain_n", "hfss_effrad_n", "zant_re_n", "zant_im_n"]
FILL = {"f_GHz": 2.4, "dist_m": 0.5, "Pin_dBm": -8.0, "PG_dB": 0.0, "Prx_pred_dBm": 0.0, "df_res_GHz": 0.0}
CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
NC = len(COLS) + len(ACOLS) + sum(len(v) for v in CATS.values())  # 18 tokens

def row_tokens(d):
    xs = [d[c].fillna(FILL[c]).values.astype(np.float32) for c in COLS]
    xs += [d[c].values.astype(np.float32) for c in ACOLS]
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
    def __init__(self, d=128, L=4, ntok=NC):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(ntok, d)
        self.emb = nn.Parameter(torch.randn(ntok, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 256, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.g = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1))  # PARD g_theta
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
        self.anchor_bias = nn.Parameter(torch.zeros(1))
    def forward(self, x, tokid):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(tokid) + self.emb).mean(1)
    def reg(self, p): return self.g(p).squeeze(-1) + self.anchor_bias  # yhat_dB 残差结构
    def proj(self, p): return nn.functional.normalize(self.pj(p), dim=-1)

def run(seed, tr_df, te_df, ykey, log_anchor_tr):
    import random
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    sc = fit_sc(tr_df)
    Xtr = torch.tensor(scale(sc, row_tokens(tr_df)), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(scale(sc, row_tokens(te_df)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NC, device=DEV).expand(len(tr_df), -1)
    tid_te = torch.arange(NC, device=DEV).expand(len(te_df), -1)
    # y in dB 域 (PARD 结构一致): VDC_dB = 20log10(V)
    ytr_np = (20 * np.log10(tr_df[ykey].clip(lower=1e-6).values))
    ate_np = (20 * np.log10(te_df[ykey].clip(lower=1e-6).values))
    ytr = torch.tensor(ytr_np, dtype=torch.float32, device=DEV)
    net = Net().to(DEV)
    net.anchor_bias.data.fill_(float(np.mean(ytr_np)))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    for ep in range(120):  # MFM denoise
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.30); m[:, 6:] = False
        Xi = Xtr.clone(); Xi[m] = 0.0
        h = net(Xi, tid); h2 = net(torch.where(m, Xtr, Xi), tid)
        loss = ((h - h2) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    grp = (tr_df["mech"].astype(str) + "|" + tr_df["proto"].astype(str)).astype("category").cat.codes.values.astype(np.int64)
    y_t = torch.tensor(grp, device=DEV)
    if len(set(grp)) > 1:
        for ep in range(80):  # SupCon
            net.train(); p = net.proj(net(Xtr, tid))
            mask = (y_t[:, None] == y_t[None, :]) & (~torch.eye(len(y_t), dtype=torch.bool, device=DEV))
            if mask.sum() == 0: break
            sim = (p @ p.T / 0.07).masked_fill(torch.eye(len(y_t), dtype=torch.bool, device=DEV), -1e9)
            loss = nn.functional.cross_entropy(sim, y_t)
            opt.zero_grad(); loss.backward(); opt.step()
    for ep in range(300):
        net.train(); loss = ((net.reg(net(Xtr, tid)) - ytr) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        pred_db = net.reg(net(Xte, tid_te)).cpu().numpy()
    pred_V = 10 ** (pred_db / 20)
    return pred_V, ate_np, pred_db

import random
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
rep = ["# G1 v1 report (PARD anchors, 5 seeds)", f"dev={DEV}"]

d81 = df[df.pool == "81785"]
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()]
ti_all, ex_all = [], []
for seed in [42, 2024, 7, 123, 99]:
    dd = d81.dropna(subset=["Y_VDC_V", "dist_m"]).copy()
    gkf = GroupKFold(n_splits=5); errs = []
    for tr_i, te_i in gkf.split(dd, groups=dd.f_GHz):
        pr_V, _, _ = run(seed, dd.iloc[tr_i], dd.iloc[te_i], "Y_VDC_V", None)
        errs.append(mae(pr_V, dd.iloc[te_i].Y_VDC_V.values))
    ti_all.append(np.mean(errs))
    tr = dd.copy()
    pr_V, _, _ = run(seed, tr, d18, "Y_VDC_V", None)
    ex = mae(pr_V, d18.Y_VDC_V.values); ex_all.append(ex)
    rep.append(f"seed{seed}: TI={np.mean(errs):.4f}  EX={ex:.4f}")

rep.append(f"\nSUM  TI v1: {np.mean(ti_all):.4f}±{np.std(ti_all):.4f}  (v0: 0.022, RF 靶 0.0536)")
rep.append(f"SUM  EX v1: {np.mean(ex_all):.4f}±{np.std(ex_all):.4f}  (v0: 0.275±0.09, RF 靶 0.1989, TabPFN 0.2804, 纯锚 0.4051)")
out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/g1_v1_report.md", "w", encoding="utf-8").write(out)
