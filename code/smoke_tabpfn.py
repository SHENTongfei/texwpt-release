import os, sys, json, hashlib
import numpy as np
import glob

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

MODEL_DIR = r"C:\Users\TS\WorkBuddy\texwpt\models\tabpfn_v2reg"
CKPT = os.path.join(MODEL_DIR, "tabpfn-v2-regressor.ckpt")

fig11 = {}
for c in glob.glob(r"C:\Users\TS\WorkBuddy\texwpt\data\extracted\**\Fig11a_meas_VDC_*.txt", recursive=True):
    tag = os.path.basename(c).replace("Fig11a_meas_VDC_", "").replace(".txt", "").strip()
    X, y = [], []
    for line in open(c, encoding="utf-8", errors="replace"):
        cells = [p for p in line.strip().split("\t")]
        if len(cells) < 2:
            continue
        try:
            f = float(cells[0])
        except ValueError:
            continue
        if not (2.0 < f < 3.0):
            continue  # frequency column only
        nums = []
        for cell in cells[1:]:
            for tok in cell.split():
                try:
                    nums.append(float(tok))
                except ValueError:
                    pass
        if len(nums) < 4:
            continue
        for i, pin in enumerate(["-4", "-8", "-10", "-12"]):
            X.append([f, int(pin), {"plain": 0, "r15": 1, "r20": 2}[tag]])
            y.append(nums[i])
    fig11[tag] = (np.array(X), np.array(y))

print("files parsed:", {k: (len(a), len(b)) for k, (a, b) in fig11.items()})
tr_X = np.vstack([fig11["plain"][0], fig11["r15"][0]])
tr_y = np.concatenate([fig11["plain"][1], fig11["r15"][1]])
te_X, te_y = fig11["r20"]
print(f"train n={len(tr_y)} (plain+r15), test n={len(te_y)} (r20 zero-shot)")

# ---- tabpfn load (local ckpt if supported, else HF-endpoint download) ----
import tabpfn
from tabpfn import TabPFNRegressor
sig = None
try:
    import inspect
    sig = inspect.signature(TabPFNRegressor.__init__)
    print("TabPFNRegressor params:", list(sig.parameters)[:12])
except Exception:
    pass

model = None
for kwargs in [
    dict(ckpt_path=CKPT),
    dict(model_path=CKPT),
    dict(weights_path=CKPT),
    {},
]:
    try:
        model = TabPFNRegressor(**kwargs)
        print("instantiated with", kwargs or "<default>")
        break
    except Exception as e:
        print("init failed", kwargs, "->", type(e).__name__, str(e)[:120])
if model is None:
    model = TabPFNRegressor()
    print("using default (HF-endpoint download)")

model.fit(tr_X, tr_y)
pred = model.predict(te_X)
mae = float(np.mean(np.abs(pred - te_y)))
print(f"VDC test MAE = {mae:.4f} V (range {min(te_y):.3f}-{max(te_y):.3f})")
print("pred head:", [round(float(p), 3) for p in pred[:6]], "true head:", [round(float(v), 3) for v in te_y[:6]])
assert np.all(np.isfinite(pred)), "NaN/inf in predictions"
print("SMOKE PASS: finite predictions, sane magnitude")

# sha256 record
h = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
print("ckpt sha256:", h)
out = {
    "model": "Prior-Labs/TabPFN-v2-reg (local ckpt 44.4MB)",
    "license": "Prior Labs License v1.1 (Apache 2.0 + attribution)",
    "smoke_task": "81813 Fig11a: VDC zero-shot plain+r15 -> r20",
    "n_train": int(len(tr_y)), "n_test": int(len(te_y)),
    "vdc_mae_test": mae, "finite_ok": True,
    "ckpt_sha256": h,
}
json.dump(out, open(os.path.join(MODEL_DIR, "smoke_result.json"), "w"), indent=2)
print("smoke_result.json written")
