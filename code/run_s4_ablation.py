# -*- coding: utf-8 -*-
"""TEXWPT S4: ablation on FINAL/I7 (leak-free features, post-AAR). 5 seeds x TI/EX.
Arms: Final(full) / -gate(v4-direct) / -I9 / -I8 / -I9-I8(=I7) / -physics / -anchor / -smithQ / -all-sim.
"""
import os, random, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from features import d81, d18, FULL, R_EST, DEV, CATS, FILLD

SEEDS = [42, 2024, 7, 123, 99]
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
t0 = time.time()
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))

def tok_matrix(d, cols):
    xs = [d[c].fillna(FILLD.get(c, 0.0)).values.astype(np.float32) for c in cols]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1), len(cols) + 7

class AblNet(nn.Module):
    def __init__(self, ntok, use_gate=True, d=96, L=3):
        super().__init__()
        self.use_gate = use_gate
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(ntok, d)
        self.emb = nn.Parameter(torch.randn(ntok, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
        self.logR = nn.Parameter(torch.tensor(float(np.log10(R_EST))))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

def run_arm(seed, tr, te, cols, use_gate, use_i9, use_i8, direct_v4=False):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    xs_tr, ntok = tok_matrix(tr, cols); xs_te, _ = tok_matrix(te, cols)
    mu = {c: (tr[c].fillna(FILLD.get(c, 0.0)).mean(), max(tr[c].fillna(FILLD.get(c, 0.0)).std(), 1e-4)) for c in cols}
    def sc(mx):
        o = mx.copy()
        for j, c in enumerate(cols): o[:, j] = (o[:, j] - mu[c][0]) / mu[c][1]
        return o
    Xtr = torch.tensor(sc(xs_tr), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(sc(xs_te), dtype=torch.float32, device=DEV)
    tid = torch.arange(ntok, device=DEV).expand(len(tr), -1)
    tidt = torch.arange(ntok, device=DEV).expand(len(te), -1)
    ytr = torch.tensor(tr.Y_VDC_V.values.astype(np.float32), device=DEV)
    net = AblNet(ntok, use_gate=use_gate).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    aDBtr = torch.tensor(tr.anchor_dB.values.astype(np.float32), device=DEV)
    aDBte = torch.tensor(te.anchor_dB.values.astype(np.float32), device=DEV)
    is81_tr = torch.tensor((tr.pool == "81785").values.astype(np.float32), device=DEV)
    is81_te = torch.tensor((te.pool == "81785").values.astype(np.float32), device=DEV)
    pin_tr = torch.tensor(tr.pin_lin_W.fillna(1e-4).values.astype(np.float32), device=DEV)
    has_eff = torch.tensor(tr.Y_eff_01.notna().values.astype(np.float32), device=DEV)
    yeff = torch.tensor(tr.Y_eff_01.fillna(0.0).values.astype(np.float32), device=DEV)
    for ep in range(100):
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(cols):] = False
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
        delta = net.head(h).squeeze(-1)
        if use_i9:
            a_tr = 10 ** ((aDBtr + is81_tr * 10.0 * (net.logR - np.log10(R_EST))) / 20)
        else:
            a_tr = 10 ** (aDBtr / 20)
        pred = delta if not use_gate else (a_tr + net.gate(h).squeeze(-1) * delta)
        if direct_v4:
            pred = net.head(h).squeeze(-1) + 0 * a_tr  # direct: pure regression from features
        loss = ((pred - ytr) ** 2).mean()
        if use_i8:
            Rl = 10 ** net.logR
            eff_pred = (pred ** 2) / (Rl * pin_tr.clamp(min=1e-6))
            loss = loss + 0.3 * (((eff_pred.clamp(0, 1) - yeff) ** 2) * has_eff).sum() / has_eff.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h = net(Xte, tidt)
        delta = net.head(h).squeeze(-1)
        if use_i9:
            a_te = 10 ** ((aDBte + is81_te * 10.0 * (net.logR - np.log10(R_EST))) / 20)
        else:
            a_te = 10 ** (aDBte / 20)
        if direct_v4:
            pred = delta
        elif use_gate:
            pred = a_te + net.gate(h).squeeze(-1) * delta
        else:
            pred = delta
    return np.clip(pred.cpu().numpy(), 0.01, 2.0)

ARMS = {
    "Final(gate+I9+I8)": (FULL, True, True, True, False),
    "I7(gate, noI9noI8)": (FULL, True, False, False, False),
    "-gate(direct)": (FULL, False, True, True, True),
    "-I9(keep gate+I8)": (FULL, True, False, True, False),
    "-I8(keep gate+I9)": (FULL, True, True, False, False),
    "-physics": ([c for c in FULL if c not in ("sq_shape", "sat_shape", "pin_off_db")], True, True, True, False),
    "-anchor": ([c for c in FULL if c not in ("anchor_V", "anchor_dB")], True, True, True, False),
    "-smithQ": ([c for c in FULL if "smith" not in c and c != "Q_proxy"], True, True, True, False),
    "-all-sim(纯数据)": (["f_GHz", "pin_avail", "df_res_GHz"], True, True, True, False),
}
rep = ["# S4 ablation on leak-free features (5 seeds)", "", "| 臂 | TI | EX |", "|---|---|---|"]
out = {}
for arm, (cols, ug, u9, u8, dv) in ARMS.items():
    ti_all, ex_all = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            errs.append(mae(run_arm(seed, d81.iloc[tr_i], d81.iloc[te_i], cols, ug, u9, u8, dv), d81.iloc[te_i].Y_VDC_V.values))
        ti_all.append(np.mean(errs))
        ex_all.append(mae(run_arm(seed, d81, d18, cols, ug, u9, u8, dv), d18.Y_VDC_V.values))
    r = (float(np.mean(ti_all)), float(np.std(ti_all)), float(np.mean(ex_all)), float(np.std(ex_all)))
    out[arm] = r
    rep.append(f"| {arm} | {r[0]:.4f}±{r[1]:.4f} | {r[2]:.4f}±{r[3]:.4f} |")
    print(f"[{arm}] TI={r[0]:.4f} EX={r[2]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(out, open(f"{ROOT}/data/qa/s4_ablation.json", "w"), indent=1)
out_s = "\n".join(rep); print(out_s)
open(f"{ROOT}/data/qa/s4_ablation.md", "w", encoding="utf-8").write(out_s)
