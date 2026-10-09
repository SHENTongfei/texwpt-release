# -*- coding: utf-8 -*-
"""TEXWPT S3: H0f baseline layer, two legs (CPU).
Leg TI: 81785 GroupKFold(by f) 5-fold -> VDC/PDC_dBm/eff
Leg EX(pre): 81785 train -> 81813 zero-shot (cross-proto) vs pure-anchor control
"""
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv")
df["pool"] = df["pool"].astype(str)

FEATS = ["f_GHz", "dist_m", "Pin_dBm", "PG_dB", "Prx_pred_dBm", "df_res_GHz",
         "mech_bent", "proto_r15", "proto_r20", "coup_body"]
def fe(d):
    x = pd.DataFrame(index=d.index)
    fills = {"f_GHz": 2.4, "dist_m": 0.5, "Pin_dBm": -8.0, "PG_dB": None, "Prx_pred_dBm": None, "df_res_GHz": 0.0}
    for c in ["f_GHz", "dist_m", "Pin_dBm", "PG_dB", "Prx_pred_dBm", "df_res_GHz"]:
        v = d[c]
        miss = v.isna().astype(int)
        if fills[c] is None:  # 派生列 = 由源列回填中性
            src = "dist_m" if c in ("PG_dB", "Prx_pred_dBm") else "f_GHz"
            v = v.fillna(d[src].map(lambda t: 0.0) if c == "df_res_GHz" else 0.0)
        else:
            v = v.fillna(fills[c])
        x[c] = v
        x[f"has_{c}"] = 1 - miss
    x["mech_bent"] = (d["mech"] == "bent").astype(int)
    x["proto_r15"] = (d["proto"] == "r15").astype(int)
    x["proto_r20"] = (d["proto"] == "r20").astype(int)
    x["coup_body"] = (d["coup"] == "body").astype(int)
    return x

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def models():
    return {
        "Ridge": Ridge(alpha=1.0),
        "RF": RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=8),
    }

try:
    from xgboost import XGBRegressor
    def xgb(): return {"XGB": XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                                           subsample=0.9, colsample_bytree=0.9, random_state=42, n_jobs=8)}
except Exception:
    from sklearn.ensemble import GradientBoostingRegressor
    def xgb(): return {"GBM": GradientBoostingRegressor(n_estimators=400, max_depth=3, learning_rate=0.05, random_state=42)}

rep = ["# S3 baseline report (2026-09-27)", ""]

# ---------- Leg TI: 81785 内 GroupKFold(by f) ----------
d81 = df[(df.pool == "81785")].copy()
for ycol, tag in [("Y_VDC_V", "VDC_V"), ("Y_PDC_W", "PDC_W"), ("Y_eff_01", "eff")]:
    dd = d81.dropna(subset=[ycol, "dist_m"]).copy()
    if tag == "PDC_W":
        dd["y"] = 10 * np.log10(dd[ycol].clip(lower=1e-13) / 1e-3)  # dBm
    else:
        dd["y"] = dd[ycol]
    X, y, g = fe(dd), dd["y"].values, dd["f_GHz"].values
    gkf = GroupKFold(n_splits=5)
    res = {k: [] for k in list(models()) + list(xgb())}
    for tr, te in gkf.split(X, y, groups=g):
        sc = StandardScaler().fit(X.iloc[tr]); Xtr, Xte = sc.transform(X.iloc[tr]), sc.transform(X.iloc[te])
        for k, mdl in {**models(), **xgb()}.items():
            mdl.fit(Xtr, y[tr]); res[k].append(mae(mdl.predict(Xte), y[te]))
    rep.append(f"**Leg TI 81785 {tag}** (n={len(dd)}, GroupKFold by f): " +
               ", ".join(f"{k}={np.mean(v):.4f}±{np.std(v):.4f}" for k, v in res.items()))

# ---------- Leg EX(pre): 81785 -> 81813 zero-shot, VDC ----------
tr = d81.dropna(subset=["Y_VDC_V", "dist_m"]).copy()
te = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()].copy()  # 主网格行(Fig11a), rot 单独
sc = StandardScaler().fit(fe(tr)); Xtr = sc.transform(fe(tr)); Xte = sc.transform(fe(te))
ytr = tr["Y_VDC_V"].values
rep.append(f"\n**Leg EX 81785->81813 VDC zero-shot** (train n={len(tr)}, test n={len(te)}):")
for k, mdl in {**models(), **xgb()}.items():
    mdl.fit(Xtr, ytr)
    rep.append(f"  {k}: MAE_V = {mae(mdl.predict(Xte), te.Y_VDC_V.values):.4f} V")

# 纯锚对照: 81813 ADS VDC 曲面(Fig5) 在 (Rin,Xin) 上的值 — 无逐文件 Zant(f) 映射时用曲面均值作保守锚
an = pd.read_csv(f"{ROOT}/data/processed/anchors.csv")
v5 = an[an.anchor_id == "81813_Fig5_ADS_VDC"]
rep.append(f"  PureAnchor(81813 Fig5 ADS VDC surface median): MAE_V = {mae(np.full(len(te), v5.y_val.median()), te.Y_VDC_V.values):.4f} V (surf median={v5.y_val.median():.4f} V, range [{v5.y_val.min():.4g},{v5.y_val.max():.4g}])")

# TabPFN (CPU)
try:
    from tabpfn import TabPFNRegressor
    ck = f"{ROOT}/models/tabpfn_v2reg/tabpfn-v2-regressor.ckpt"
    m = TabPFNRegressor(model_path=ck, device="cpu")
    m.fit(fe(tr).values, ytr)
    rep.append(f"  TabPFN: MAE_V = {mae(m.predict(fe(te).values), te.Y_VDC_V.values):.4f} V")
except Exception as e:
    rep.append(f"  TabPFN SKIP: {e}")

out = "\n".join(rep)
open(f"{ROOT}/data/qa/baseline_report.md", "w", encoding="utf-8").write(out)
print(out)
