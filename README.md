# TEXWPT — Transferable Estimation of X-Prototype Wireless Power Transfer

Companion release for the manuscript **"Physics-Anchored Gated Transformers Recover the Physical Drivers of Rectified Voltage for Transferable, Cross-Prototype Wireless Power Transfer"** (model: TEXWPT, IEEE Transactions on Artificial Intelligence, under preparation).

TEXWPT is a physics-anchored gated transformer that predicts the rectified output voltage of a wireless power transfer (WPT) front-end from physical and operational inputs, and transfers to unseen prototypes and an independent laboratory platform without retraining.

## Headline results

- Lowest cross-prototype (zero-shot) mean absolute error among 14 reference methods: **0.116 V** (external-1).
- Outperforms the strongest tabular baselines by roughly **2x** on the primary engineering metric.
- Blind cross-platform probe (independent magnetic near-field lab, 865.5 MHz): **tied for first, transferable**.
- All results reproduce from **public data only**; no private measurements are required.

## Repository layout

```
README.md                    this file
SHA256_LEDGER.txt            checksum ledger of verified data files
benchmark/                   primary result tables (CSV + Markdown)
code/                        core dataset/feature/baseline/inference scripts
configs/                     model configurations + sample training state
data_processed/              verified, processed public data (subset)
supplementary/               QA audits, gate reports, extended analyses
```

## Datasets (all public)

| Pool | Frequency | Role |
|---|---|---|
| Oviedo textile CP rectifier (woven) | 2.44 GHz | Primary; training + in-pool evaluation |
| Oviedo textile CP rectifier (2 variants) | 2.4 GHz | External-1; cross-prototype zero-shot |
| Magnetic near-field (independent lab) | 865.5 MHz | External-2; blind cross-platform probe |
| ADS harmonic-balance + HFSS antenna traces | - | Physics anchor only (excluded from measured training) |

## Reproduction

```bash
python code/build_dataset.py     # assemble the processed pools
python code/features.py          # physics-informed feature construction
python code/run_baselines.py     # 14 reference estimators + TEXWPT
```

Configurations in `configs/` match the released training state. The `supplementary/` directory carries the QA gate reports and extended audits.

## Scope of this release

Per the open-release convention, this repository intentionally excludes the full manuscript text, full-resolution figures, and full training weights. It ships a verified data subset, benchmark tables, core code, configurations, and a sample checkpoint sufficient to reproduce the headline results.

## License and attribution

Data pools are used under their published research-reuse licenses (Oviedo University DSpace; independent-lab open dataset). Code is released for research reuse.
