# -*- coding: utf-8 -*-
"""TEXWPT verdict stats: combine ours + TabPFN + anchor + GP per-fold data -> Wilcoxon + Holm."""
import json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
ours = json.load(open(f"{ROOT}/data/qa/verdict_ours.json"))
others = json.load(open(f"{ROOT}/data/qa/verdict_others.json"))

SEEDS = [42, 2024, 7, 123, 99, 555, 777, 31337]
def flatten_fold(seed, folds): return [f"seed{seed}_fold{i}" for i in range(len(folds))]

rep = ["# TEXWPT 终裁裁决（8-seed，折级配对 Wilcoxon + Holm）", ""]
rows = []
for cfg in ["I7", "Final"]:
    ti = np.array(ours[cfg]["TI_folds"]); ex = np.array(ours[cfg]["EX"])
    rows.append((cfg, ti.mean(), ti.std(), ex.mean(), ex.std()))
for name, d in others.items():
    ti = np.array(d["TI_folds"]); ex = np.array(d["EX"])
    rows.append((name, ti.mean(), ti.std(), ex.mean(), ex.std()))
rep.append("| 配置 | TI mean±std | EX mean±std |"); rep.append("|---|---|---|")
for n, tm, ts, em, es in sorted(rows, key=lambda r: r[3]):
    rep.append(f"| {n} | {tm:.4f}±{ts:.4f} | {em:.4f}±{es:.4f} |")
rep.append("")

# EX 腿裁决（主战轴）：主模型 vs 各对照（配对 by seed）
rep.append("## EX 腿（跨原型零样本主战轴）配对裁决")
best = min(["I7", "Final"], key=lambda c: np.mean(ours[c]["EX"]))
rep.append(f"主模型（EX 最优）= **{best}**")
tests = []
for name, d in others.items():
    for cfg in ["I7", "Final"]:
        a = np.array(ours[cfg]["EX"]); b = np.array(d["EX"])
        diff = b - a  # 正 = 对照更差 = 我们赢
        try:
            stat, p = wilcoxon(a, b)
        except Exception:
            p = 1.0
        tests.append((f"{cfg} vs {name}", float(np.mean(a - b)), float(p), int((a < b).sum()), len(a)))
# I7 vs Final 自身
a = np.array(ours["Final"]["EX"]); b = np.array(ours["I7"]["EX"])
stat, p = wilcoxon(a, b)
tests.append(("Final vs I7", float(np.mean(a - b)), float(p), int((a < b).sum()), len(a)))
# Holm
mt = sorted(tests, key=lambda t: t[2])
m = len(mt); holm = [min(1.0, p * (m - i)) for i, p in enumerate([t[2] for t in mt])]
holm_map = {t[0]: h for t, h in zip(mt, holm)}
rep.append("| 对比 | ΔMAE(我们-对照, 负=赢) | p (Wilcoxon) | p (Holm) | 胜/总 seeds |")
rep.append("|---|---|---|---|---|")
for name, delta, p, w, n in tests:
    sig = "✓" if holm_map[name] < 0.05 else "ns"
    rep.append(f"| {name} | {delta:+.4f} | {p:.4f} | {holm_map[name]:.4f} {sig} | {w}/{n} |")

# TI 腿（折级配对，8 seeds × 5 folds = 40 对）
rep.append(""); rep.append("## TI 腿（折级配对，n=40）")
for name, d in others.items():
    if "TI_folds" not in d: continue
    for cfg in ["I7", "Final"]:
        a = np.concatenate(ours[cfg]["TI_folds"]); b = np.concatenate(d["TI_folds"])
        try: stat, p = wilcoxon(a, b)
        except Exception: p = 1.0
        rep.append(f"- {cfg} vs {name}: Δ={np.mean(a-b):+.4f} p={p:.4f}")
out = "\n".join(rep)
open(f"{ROOT}/data/qa/final_verdict.md", "w", encoding="utf-8").write(out)
print(out)
