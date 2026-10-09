# -*- coding: utf-8 -*-
"""TEXWPT FINAL: FULL features + I7 gate + I9 learnable R_eff + I8 physics closure. 5 seeds x TI/EX.
Compare: I7(EX 0.1173), PureAnchor(0.1242), GP(0.1992), TabPFN(0.2444), TabPFN-TI(0.0196).
"""
import os, random, time, json
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import GroupKFold
from features import d81, d18, FULL, R_EST, DEV, CATS, FILLD  # 已验证构造

ROOT = "C:/Users/TS/WorkBuddy/texwpt"
SEEDS = [42, 2024, 7, 123, 99]
t0 = time.time()
def mae(a, b): return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))
NTOK = len(FULL) + 7

def tok_matrix(d):
    xs = [d[c].fillna(FILLD[c]).values.astype(np.float32) for c in FULL]
    xs += [(d[k].values == cat).astype(np.float32) for k, cats in CATS.items() for cat in cats]
    return np.stack(xs, axis=1)

class FinalNet(nn.Module):
    def __init__(self, d=96, L=3):
        super().__init__()
        self.inp = nn.Linear(1, d); self.tok = nn.Embedding(NTOK, d)
        self.emb = nn.Parameter(torch.randn(NTOK, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, 4, 192, batch_first=True, dropout=0.1)
        self.tr = nn.TransformerEncoder(enc, L)
        self.head = nn.Sequential(nn.Linear(d, 48), nn.GELU(), nn.Linear(48, 1))
        self.gate = nn.Sequential(nn.Linear(d, 16), nn.GELU(), nn.Linear(16, 1), nn.Sigmoid())
        self.pj = nn.Sequential(nn.Linear(d, 32), nn.GELU(), nn.Linear(32, 16))
        self.logR = nn.Parameter(torch.tensor(float(np.log10(R_EST))))  # I9

    def forward(self, x, t):
        return self.tr(self.inp(x.unsqueeze(-1)) + self.tok(t) + self.emb).mean(1)

def run_final(seed, tr, te):
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
    net = FinalNet().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-2)
    ytr = torch.tensor(tr.Y_VDC_V.values.astype(np.float32), device=DEV)
    pin_tr = torch.tensor(tr.pin_lin_W.fillna(1e-4).values.astype(np.float32), device=DEV)
    has_eff = torch.tensor(tr.Y_eff_01.notna().values.astype(np.float32), device=DEV)
    yeff = torch.tensor(tr.Y_eff_01.fillna(0.0).values.astype(np.float32), device=DEV)
    for ep in range(100):  # MFM
        net.train(); m = (torch.rand(Xtr.shape, device=DEV) < 0.3); m[:, len(FULL):] = False
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
        a_tr = 10 ** ((aDBtr + is81_tr * 10.0 * (net.logR - np.log10(R_EST))) / 20)  # I9
        pred = a_tr + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)              # I7
        loss = ((pred - ytr) ** 2).mean()
        Rl = 10 ** net.logR
        eff_pred = (pred ** 2) / (Rl * pin_tr.clamp(min=1e-6))                        # I8
        loss = loss + 0.3 * (((eff_pred.clamp(0, 1) - yeff) ** 2) * has_eff).sum() / has_eff.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h = net(Xte, tidt)
        a_te = 10 ** ((aDBte + is81_te * 10.0 * (net.logR - np.log10(R_EST))) / 20)
        pred = a_te + net.gate(h).squeeze(-1) * net.head(h).squeeze(-1)
    return np.clip(pred.cpu().numpy(), 0.01, 2.0)

rep = [f"# FINAL (FULL+I7 gate+I9 R_eff+I8 closure) R_EST={R_EST/1000:.2f}k dev={DEV}", ""]
ti_all, ex_all = [], []
for seed in SEEDS:
    gkf = GroupKFold(n_splits=5); errs = []
    for tr_i, te_i in gkf.split(d81, groups=d81.f_GHz):
        errs.append(mae(run_final(seed, d81.iloc[tr_i], d81.iloc[te_i]), d81.iloc[te_i].Y_VDC_V.values))
    ti = float(np.mean(errs))
    ex = mae(run_final(seed, d81, d18), d18.Y_VDC_V.values)
    ti_all.append(ti); ex_all.append(ex)
    print(f"seed{seed}: TI={ti:.4f} EX={ex:.4f} ({time.time()-t0:.0f}s)", flush=True)
rep.append(f"FINAL TI: {np.mean(ti_all):.4f}±{np.std(ti_all):.4f}   (TabPFN 0.0196 / SVR 0.0294 / I7 0.0357)")
rep.append(f"FINAL EX: {np.mean(ex_all):.4f}±{np.std(ex_all):.4f}   (I7 0.1173 / anchor 0.1242 / GP 0.1992 / TabPFN 0.2444)")
out = "\n".join(rep); print(out)
open(f"{ROOT}/data/qa/final_config_report.md", "w", encoding="utf-8").write(out)
json.dump({"TI": [float(x) for x in ti_all], "EX": [float(x) for x in ex_all]}, open(f"{ROOT}/data/qa/final_config.json", "w"), indent=1)
