# -*- coding: utf-8 -*-
"""TEXWPT verdict (ours): I7 vs Final, 8 seeds, per-fold TI + EX MAE -> json for stats."""
import os, random, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from features import d81, d18, FULL, R_EST, DEV, CATS, FILLD

SEEDS = [42, 2024, 7, 123, 99, 555, 777, 31337]
ROOT = "C:/Users/TS/WorkBuddy/texwpt"
t0 = time.time()
NTOK = len(FULL) + 7

def tok_matrix(d):
    xs = [d[c].fillna(FILLD[c]).values.astype(np.float32) for c in FULL]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)

class VNet(nn.Module):
    def __init__(self, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(NTOK, d)
        self.emb = nn.Parameter(torch.randn(NTOK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
        self.logR = nn.Parameter(torch.tensor(float(np.log10(R_EST))))
    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

def run_cfg(seed, tr, te, use_i9, use_i8):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    mu_s = {c: (tr[c].fillna(FILLD[c]).mean(), max(tr[c].fillna(FILLD[c]).std(), 1e-4)) for c in FULL}
    def sc(mx):
        o = mx.copy()
        for j, c in enumerate(FULL): o[:, j] = (o[:, j] - mu_s[c][0]) / mu_s[c][1]
        return o
    Xtr = torch.tensor(sc(tok_matrix(tr)), dtype=torch.float32, device=DEV)
    Xte = torch.tensor(sc(tok_matrix(te)), dtype=torch.float32, device=DEV)
    tid = torch.arange(NTOK, device=DEV).expand(len(tr), -1)
    tidt = torch.arange(NTOK, device=DEV).expand(len(te), -1)
    is81_tr = torch.tensor((tr.pool == "81785").values.astype(np.float32), device=DEV)
    is81_te = torch.tensor((te.pool == "81785").values.astype(np.float32), device=DEV)
    aDBtr = torch.tensor(tr.anchor_dB.values.astype(np.float32), device=DEV)
    aDBte = torch.tensor(te.anchor_dB.values.astype(np.float32), device=DEV)
    net = VNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    ytr = torch.tensor(tr.Y_VDC_V.values.astype(np.float32), device=DEV)
    pin_tr = torch.tensor(tr.pin_lin_W.fillna(1e-4).values.astype(np.float32), device=DEV)
    has_eff = torch.tensor(tr.Y_eff_01.notna().values.astype(np.float32), device=DEV)
    yeff = torch.tensor(tr.Y_eff_01.fillna(0.0).values.astype(np.float32), device=DEV)
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
        if use_i9:
            a_tr = 10 ** ((aDBtr + is81_tr * 10.0 * (net.logR - np.log10(R_EST))) / 20)
        else:
            a_tr = 10 ** (aDBtr / 20)
        pred = a_tr + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
        loss = ((pred - ytr) ** 2).mean()
        if use_i8:
            Rl = 10 ** net.logR
            eff_pred = (pred ** 2) / (Rl * pin_tr.clamp(min=1e-6))
            loss = loss + 0.3 * (((eff_pred.clamp(0, 1) - yeff) ** 2) * has_eff).sum() / has_eff.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h = net(Xte, tidt)
        if use_i9:
            a_te = 10 ** ((aDBte + is81_te * 10.0 * (net.logR - np.log10(R_EST))) / 20)
        else:
            a_te = 10 ** (aDBte / 20)
        pred = a_te + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
    return np.clip(pred.cpu().numpy(), 0.01, 2.0)

def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
out = {}
for cfg, (u9, u8) in [("I7", (False, False)), ("Final", (True, True))]:
    ti_folds, ex = [], []
    for seed in SEEDS:
        gkf = GroupKFold(n_splits=5); errs = []
        for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
            errs.append(mae(run_cfg(seed, d81.iloc[tr_i], d81.iloc[te_i], u9, u8), d81.iloc[te_i].Y_VDC_V.values))
        ti_folds.append(errs)
        ex.append(mae(run_cfg(seed, d81, d18, u9, u8), d18.Y_VDC_V.values))
        print(f"[{cfg}] seed{seed}: TI={errs} EX={ex[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
    out[cfg] = {"TI_folds": ti_folds, "EX": ex}
json.dump(out, open(f"{ROOT}/data/qa/verdict_ours.json", "w"), indent=1)
print("OURS VERDICT DATA DONE")
