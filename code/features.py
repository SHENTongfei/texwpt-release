"""TEXWPT shared feature construction (verified logic from run_p1.py)."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
import time
import numpy as np, pandas as pd, torch, torch.nn as nn

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(f"{ROOT}/data/processed/measured_long.csv")
df["pool"] = df["pool"].astype(str)
an = pd.read_csv(f"{ROOT}/data/processed/anchors.csv")

g81 = an[an.anchor_id == "81785_Fig4c_gain_air"].sort_values("x1")
ads18v = an[an.anchor_id == "81813_Fig5_ADS_VDC"]
ads81e = an[an.anchor_id == "81785_Fig2_ADS_eff"]
r3 = an[an.anchor_id == "81813_Fig3rect_ADS_VDC"].merge(
    an[an.anchor_id == "81813_Fig3rect_ADS_eff"], on=["x1"], suffixes=("_v", "_e"))
R_EST = float(np.median((r3.y_val_v ** 2) / (10 ** (-40 / 10) * r3.y_val_e)))
z18g = an[an.anchor_id == "81813_Fig6_HFSS_Z"]
Z18R = z18g[z18g.x2 == 0].sort_values("x1")
Z18I = z18g[z18g.x2 == 1].sort_values("x1")
z81g = an[an.anchor_id == "81785_Fig3air_HFSS_Z"]
Z81R = z81g[z81g.x2 == 0].sort_values("x1")
Z81I = z81g[z81g.x2 == 3].sort_values("x1")

lam = 0.125
PG = 20 * np.log10(lam / (4 * np.pi * df["dist_m"].clip(lower=0.05).fillna(0.5)))
gr81 = np.interp(df["f_GHz"], g81.x1.values, g81.y_val.values)
df["pin_avail"] = np.where(df.pool == "81785", 27.0 + gr81 + PG, df["Pin_dBm"])
df["pin_lin_W"] = 10 ** ((df["pin_avail"] - 30) / 10)

def nn2(ads, R, X):
    """AAR 二刀整改: 最近邻→双线性插值+线性外推 (规则网格 5Ω 步长).
    旧实现出界点被钳边缘常数 (81785 Z_re∈[0.8,21] vs 网格[5,50] 大量出界) → 锚形状失真."""
    from scipy.interpolate import RegularGridInterpolator
    x1s = np.sort(ads.x1.unique()); x2s = np.sort(ads.x2.unique())
    grid = np.full((len(x1s), len(x2s)), np.nan)
    pos = {v: i for i, v in enumerate(x1s)}
    pos2 = {v: i for i, v in enumerate(x2s)}
    for r in ads.itertuples():
        grid[pos[r.x1], pos2[r.x2]] = r.y_val
    # 填 NaN (网格洞) 用最近邻
    mask = np.isnan(grid)
    if mask.any():
        from scipy.interpolate import NearestNDInterpolator
        pts = np.argwhere(~mask)
        vals = grid[~mask]
        fll = NearestNDInterpolator(pts, vals)
        for i, j in np.argwhere(mask):
            grid[i, j] = fll(i, j)
    itp = RegularGridInterpolator((x1s, x2s), grid, method="linear", bounds_error=False, fill_value=None)
    return float(itp([[R, X]])[0])  # fill_value=None = 线性外推出界

zr18 = np.interp(df["f_GHz"], Z18R.x1.values, Z18R.y_val.values)
zi18 = np.interp(df["f_GHz"], Z18I.x1.values, Z18I.y_val.values)
zr81 = np.interp(df["f_GHz"], Z81R.x1.values, Z81R.y_val.values)
zi81 = np.interp(df["f_GHz"], Z81I.x1.values, Z81I.y_val.values)
ads_vdc18 = np.array([nn2(ads18v, a, b) for a, b in zip(zr18, zi18)])
ads_eff81 = np.array([nn2(ads81e, a, b) for a, b in zip(zr81, zi81)])
df["pin_off_db"] = df["pin_avail"].values + 10.0
df["anchor_V"] = np.where(df.pool == "81785",
                          np.sqrt(df["pin_lin_W"] * ads_eff81 * R_EST),
                          ads_vdc18 * 10 ** (df["pin_off_db"] / 20))
df.loc[df.anchor_V <= 0.001, "anchor_V"] = 0.001
df["anchor_dB"] = 20 * np.log10(df["anchor_V"])

for pool, ZR, ZI in [("81785", Z81R, Z81I), ("81813", Z18R, Z18I)]:
    m = df.pool == pool
    zc_r = float(np.interp(2.4, ZR.x1.values, ZR.y_val.values))
    zc_i = float(np.interp(2.4, ZI.x1.values, ZI.y_val.values))
    zr = np.interp(df.loc[m, "f_GHz"], ZR.x1.values, ZR.y_val.values)
    zi = np.interp(df.loc[m, "f_GHz"], ZI.x1.values, ZI.y_val.values)
    dz = zr - zc_r + 1j * (zi - zc_i)
    df.loc[m, "smith_ang"] = np.angle(dz)
    curve_max = float(np.abs(ZR.y_val.values - zc_r + 1j * (ZI.x1.values * 0 + ZI.y_val.values - zc_i)).max())
    df.loc[m, "smith_norm"] = np.abs(dz) / max(curve_max, 1e-6)
df["smith_ang"] = df["smith_ang"].fillna(0.0)
df["smith_norm"] = df["smith_norm"].fillna(0.0)

# AAR R1 整改 P0-2: f_res 改仿真侧 (HFSS 阻抗虚部过零 = 谐振), 不接触任何实测标签
fres_map = {}
for pool, ZR, ZI in [("81785", Z81R, Z81I), ("81813", Z18R, Z18I)]:
    fi = ZR.x1.values
    zmag = np.sqrt(ZR.y_val.values ** 2 + ZI.y_val.values ** 2)
    band = (fi >= 2.3) & (fi <= 2.5)  # 数据频带内
    idx = np.where(band)[0]
    fres_map[pool] = float(fi[idx[np.argmin(zmag[idx])]])  # 带内 |Z| 最小 = 串联谐振
df["f_res"] = df["pool"].map(fres_map)
df["df_res_GHz"] = df["f_GHz"] - df["f_res"]

# AAR R1 整改 P0-1: Q_proxy 改仿真侧 (ADS 效率轨迹峰宽), 不接触实测标签
qmap = {}
for pool, ads, ZR, ZI in [("81785", ads81e, Z81R, Z81I), ("81813", ads18v, Z18R, Z18I)]:
    fr = np.sort(df.loc[df.pool == pool, "f_GHz"].unique())
    traj_eff = []
    for f in fr:
        zr = float(np.interp(f, ZR.x1.values, ZR.y_val.values))
        zi = float(np.interp(f, ZI.x1.values, ZI.y_val.values))
        traj_eff.append(nn2(ads, zr, zi))
    traj_eff = np.array(traj_eff)
    pk = int(np.argmax(traj_eff))
    half = traj_eff[pk] / 2
    above = np.where(traj_eff >= half)[0]
    fwhm = max(fr[above.max()] - fr[above.min()], 0.01)
    qmap[pool] = float(fr[pk] / fwhm)
df["Q_proxy"] = df["pool"].map(qmap)
df["sq_shape"] = 10 ** (df["pin_off_db"] / 20)
df["sat_shape"] = 1 / (1 + np.exp(-(df["pin_avail"] + 8) / 2))

d81 = df[(df.pool == "81785") & df.Y_VDC_V.notna()].copy()
d18 = df[(df.pool == "81813") & df.Y_VDC_V.notna() & df.rot_deg.isna()].copy()

FULL = ["f_GHz", "pin_avail", "pin_off_db", "df_res_GHz", "anchor_V", "anchor_dB",
        "sq_shape", "sat_shape", "smith_ang", "smith_norm", "Q_proxy"]
CATS = {"mech": ["plain", "bent"], "proto": ["plain", "r15", "r20"], "coup": ["air", "body"]}
FILLD = {"f_GHz": 2.4, "pin_avail": -10.0, "pin_off_db": 0.0, "df_res_GHz": 0.0,
         "anchor_V": 0.15, "anchor_dB": -16.0, "sq_shape": 1.0, "sat_shape": 0.5,
         "smith_ang": 0.0, "smith_norm": 0.0, "Q_proxy": 0.0}

def mae(a, b):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
