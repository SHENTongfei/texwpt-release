# -*- coding: utf-8 -*-
"""TEXWPT S2: build unified long-table dataset from Oviedo 81785/81813 (+MHN probe later).
S1 口径: 实测入训练 = 81785 Fig6a/6b/7 (Fig8 存疑层) + 81813 Fig11a/11b/12/13; 仿真仅锚 = 其余.
单位归一: V->V, P->W, eff->0-1. Fig13 readme 声明 V 实为 mV (STOP-AND-REPORT #7).
"""
import os, re, math
import pandas as pd, numpy as np

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
D81785 = f"{ROOT}/data/extracted/oviedo_81785/datos repositorio"
D81813 = f"{ROOT}/data/extracted/oviedo_81813/datos_repositorio"
OUT = f"{ROOT}/data/processed"; os.makedirs(OUT, exist_ok=True)
EIRP_dBm = 27.0
DISTS = [0.3, 0.5, 0.65, 0.8]  # readme 权威; 表头 V(0.6m) 系笔误

def read_tab(path, skip_pct=True):
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.rstrip("\r\n")
            if skip_pct and ln.startswith("%"): continue
            if not ln.strip(): continue
            rows.append(ln.split("\t"))
    return rows

def read_ws(path):
    """HFSS/ANSOFT 导出: % 注释头 + 空格分隔数据行."""
    rows = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        ln = ln.strip()
        if not ln or ln.startswith("%"): continue
        rows.append(ln.split())
    return rows

def fnum(x):
    try: return float(x)
    except: return np.nan

rows_measured = []
def add(**kw): rows_measured.append(kw)

# ---------- 81785 Fig6a(V,mV) / Fig6b(P,W) / Fig7(eff 0-1) × {plain,bent} ----------
cell = {}  # (f, dist, mech) -> dict
for tag, col, conv in [("Fig6a", "VDC", 1e-3), ("Fig6b", "PDC", 1.0), ("Fig7", "eff", 1.0)]:
    for mech in ["plain", "bent"]:
        p = f"{D81785}/{tag}_{mech}.txt"
        rr = read_tab(p)
        data = rr[1:] if not re.match(r"[\d.]", rr[0][0]) else rr  # 跳表头
        for r in data:
            f = fnum(r[0])
            if not (2.0 < f < 3.0): continue
            for i, d in enumerate(DISTS):
                v = fnum(r[1 + i]) * conv
                cell.setdefault((round(f, 3), d, mech), {})[col] = v
for (f, d, mech), y in cell.items():
    add(pool="81785", proto="plain", mech=mech, coup="air", f_GHz=f, Pin_dBm=np.nan,
        dist_m=d, rot_deg=np.nan, case_id=1, R_load_kOhm=np.nan,
        Y_VDC_V=y.get("VDC"), Y_PDC_W=y.get("PDC"), Y_eff_01=y.get("eff"),
        file=f"81785_{mech}", measured_flag=1)

# ---------- 81813 Fig11a(VDC,V)/Fig11b(PDC,uW)/Fig12(eff,%) × {plain,r15,r20} ----------
PINS = [-4.0, -8.0, -10.0, -12.0]
cell8 = {}
for tag, col, conv in [("Fig11a_meas_VDC", "VDC", 1.0), ("Fig11b_meas_PDC", "PDC", 1e-6), ("Fig12_meas_eff", "eff", 1e-2)]:
    for proto in ["plain", "r15", "r20"]:
        p = f"{D81813}/{tag}_{proto}.txt"
        if not os.path.exists(p): continue
        rr = read_tab(p)
        data = rr[1:]
        for r in data:
            f = fnum(r[0])
            if not (2.0 < f < 3.0): continue
            ncol = len([x for x in r[1:] if x.strip() != ""])  # r15/r20 eff 仅 2 列
            pins = PINS if ncol >= 4 else [-4.0, -10.0]
            for i, pin in enumerate(pins):
                v = fnum(r[1 + i]) * conv
                cell8.setdefault((round(f, 3), pin, proto), {})[col] = v
for (f, pin, proto), y in cell8.items():
    add(pool="81813", proto=proto, mech="plain", coup="air", f_GHz=f, Pin_dBm=pin,
        dist_m=np.nan, rot_deg=np.nan, case_id=1, R_load_kOhm=np.nan,
        Y_VDC_V=y.get("VDC"), Y_PDC_W=y.get("PDC"), Y_eff_01=y.get("eff"),
        file=f"81813_{proto}", measured_flag=1)

# ---------- 81813 Fig13 RA (ang × {plain, r20_1, r20_2, r15_1, r15_2}), mV->V ----------
rr = read_tab(f"{D81813}/Fig13_meas_RA.txt")
for r in rr[1:]:
    ang = fnum(r[0])
    if ang is None or (isinstance(ang, float) and math.isnan(ang)): continue
    for j, (proto, case) in enumerate([("plain", 1), ("r20", 1), ("r20", 2), ("r15", 1), ("r15", 2)]):
        v = fnum(r[1 + j])
        if pd.isna(v): continue
        add(pool="81813", proto=proto, mech="plain", coup="air", f_GHz=2.4, Pin_dBm=np.nan,
            dist_m=np.nan, rot_deg=ang, case_id=case, R_load_kOhm=np.nan,
            Y_VDC_V=v * 1e-3, Y_PDC_W=np.nan, Y_eff_01=np.nan, file="81813_Fig13_RA", measured_flag=1)

df = pd.DataFrame(rows_measured)

# ---------- 物理先验特征 ----------
df["PG_dB"] = 20 * np.log10(0.125 / (4 * np.pi * df["dist_m"].clip(lower=0.05)))  # λ=0.125m @2.4GHz; rot-only 行 dist=NaN
df["Prx_pred_dBm"] = EIRP_dBm + 2.0 + df["PG_dB"]  # 假设 Gr=2dBi, ex-ante 声明
df["PDC_dBm"] = 10 * np.log10(df["Y_PDC_W"].clip(lower=1e-13) / 1e-3)
df["friis_resid_dB"] = df["PDC_dBm"] - df["Prx_pred_dBm"]  # 半物理残差(仅 81785 有 dist)
df["VDC_dB"] = 20 * np.log10(df["Y_VDC_V"].clip(lower=1e-6))

# AAR R1 隔离: 旧版 f_res 从实测 V 峰提取 = 标签泄漏, 已弃用(features.py 仿真侧为准)
# 保留列名改为 LEAKY 前缀, 防未来脚本误用
fres = {}
for (pool, mech), g in df.dropna(subset=["Y_VDC_V"]).groupby(["pool", "mech"]):
    gg = g.groupby("f_GHz")["Y_VDC_V"].mean()
    fres[(pool, mech)] = gg.idxmax()
df["f_res_LEAKY_DO_NOT_USE"] = [fres.get((p, m), np.nan) for p, m in zip(df["pool"], df["mech"])]
# df_res_GHz 由 features.py 仿真侧计算
df["cos2_rot"] = np.cos(np.deg2rad(df["rot_deg"].fillna(0))) ** 2

df.to_csv(f"{OUT}/measured_long.csv", index=False)

# ---------- 锚层 ----------
an = []
def anrow(**kw): an.append(kw)
# ADS 曲面
for src, path, ycol, conv in [
    ("81785_Fig2_ADS_eff", f"{D81785}/Fig2.txt", "eff", 1e-2),  # 值域实测 0.004-61.6 = 百分数 (S2 复核修正)
    ("81813_Fig4_ADS_eff", f"{D81813}/Fig4_eff_vs_Zant.txt", "eff", 1e-2),
    ("81813_Fig5_ADS_VDC", f"{D81813}/Fig5_VDC_vs_Zant.txt", "VDC", 1.0)]:
    for r in read_tab(path):
        if len(r) < 3: continue
        Rin, Xin, y = fnum(r[0]), fnum(r[1]), fnum(r[2])
        if pd.isna(Rin) or pd.isna(y): continue
        anrow(anchor_id=src, x1=Rin, x2=Xin, y_name=ycol, y_val=y * conv, note="ADS Harmonic Balance")
# 81813 Fig3_rect (ADS): R(kOhm) -> eff(%), VDC(V)
rr = read_tab(f"{D81813}/Fig3_rect_performance.txt")
for r in rr[1:]:
    R, eff, vdc = fnum(r[0]), fnum(r[1]), fnum(r[2])
    if pd.isna(R): continue
    anrow(anchor_id="81813_Fig3rect_ADS_eff", x1=R, x2=np.nan, y_name="eff", y_val=eff * 1e-2, note="ADS HB, R sweep")
    anrow(anchor_id="81813_Fig3rect_ADS_VDC", x1=R, x2=np.nan, y_name="VDC", y_val=vdc, note="ADS HB, R sweep")
# HFSS 特性曲线
for src, path, cols in [
    ("81785_Fig3air_HFSS_Z", f"{D81785}/Fig3_air.txt", None), ("81785_Fig3body_HFSS_Z", f"{D81785}/Fig3_body.txt", None),
    ("81813_Fig6_HFSS_Z", f"{D81813}/Fig6_Zant.txt", None),
    ("81785_Fig4c_eff_air", f"{D81785}/Fig4c_eff_rad_simu_air.txt", None), ("81785_Fig4c_eff_body", f"{D81785}/Fig4c_eff_rad_simu_body.txt", None),
    ("81785_Fig4c_gain_air", f"{D81785}/Fig4c_max_gain_simu_air.txt", None),
    ("81813_Fig7_pat", f"{D81813}/Fig7_simu_rad_pattern.txt", None), ("81813_Fig8_AR", f"{D81813}/Fig8_simu_AR.txt", None),
    ("81785_Fig4a_pat_air", f"{D81785}/Fig4a_air.txt", None), ("81785_Fig4b_AR_air", f"{D81785}/Fig4b_AR_air.txt", None)]:
    for r in read_ws(path):
        vals = [fnum(x) for x in r]
        vals = [v for v in vals if not (isinstance(v, float) and pd.isna(v))]
        if len(vals) < 2: continue
        for j, v in enumerate(vals[1:]):
            anrow(anchor_id=src, x1=vals[0], x2=j, y_name="curve", y_val=v, note="HFSS/ANSOFT export")
pd.DataFrame(an).to_csv(f"{OUT}/anchors.csv", index=False)

# ---------- 存疑层: 81785 Fig8 (readme: 归一化功率 dBm; 列名: AR dB) ----------
fig8 = []
rr = read_tab(f"{D81785}/Fig8.txt")
hdr = rr[0]
for r in rr[1:]:
    ang = fnum(r[0])
    for j in range(1, len(r)):
        v = fnum(r[j])
        if pd.isna(v): continue
        m = re.match(r"AR\(([\d.]+) GHz, (\w+)\)", str(hdr[j]).strip()) if j < len(hdr) else None
        ff, mech = (float(m.group(1)), m.group(2)) if m else (np.nan, "?")
        fig8.append(dict(ang_deg=ang, f_GHz=ff, mech=mech, value_dB=v,
                         readme_claim="normalized measured power dBm", header_claim="AR dB", status="CONFLICT->supplement only"))
pd.DataFrame(fig8).to_csv(f"{OUT}/fig8_conflict_supplement.csv", index=False)

# ---------- QA 报告 ----------
lines = ["# dataset QA (S2, 2026-09-27)", ""]
lines.append(f"measured_long rows = {len(df)}")
for (pool, proto), g in df.groupby(["pool", "proto"]):
    lines.append(f"  {pool}/{proto}: {len(g)} rows, VDC n={g.Y_VDC_V.notna().sum()}, PDC n={g.Y_PDC_W.notna().sum()}, eff n={g.Y_eff_01.notna().sum()}, rot n={g.rot_deg.notna().sum()}")
lines.append(f"\nY ranges: VDC [{df.Y_VDC_V.min():.4g}, {df.Y_VDC_V.max():.4g}] V; PDC [{df.Y_PDC_W.min():.3g}, {df.Y_PDC_W.max():.3g}] W; eff [{df.Y_eff_01.min():.3g}, {df.Y_eff_01.max():.3g}]")
lines.append(f"f_res extracted: {fres}")
bad = df[(df.Y_VDC_V.notna() & ((df.Y_VDC_V <= 0) | (df.Y_VDC_V > 3))) |
         (df.Y_eff_01.notna() & ((df.Y_eff_01 <= 0) | (df.Y_eff_01 > 1)))]
lines.append(f"physical-range violations: {len(bad)} (expect 0)")
lines.append(f"anchors rows = {len(an)}; fig8 conflict rows = {len(fig8)} (supplement only)")
rep = "\n".join(lines)
open(f"{ROOT}/data/qa/dataset_qa.md", "w", encoding="utf-8").write(rep)
print(rep)
