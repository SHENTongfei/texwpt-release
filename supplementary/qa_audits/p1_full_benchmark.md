# P1/P2 full benchmark (2026-09-27, 477s)

| 模型 | TI MAE_V | EX MAE_V |
|---|---|---|
| TEXWPT-I7(gated-res) | 0.0357±0.0041 | 0.1173±0.0239 |
| PureAnchor(sim) | 0.5114±0.0000 | 0.1242±0.0000 |
| GP | 0.0385±0.0000 | 0.1992±0.0000 |
| SVR | 0.0294±0.0000 | 0.2134±0.0000 |
| KNN | 0.0643±0.0000 | 0.2219±0.0000 |
| XGBoost | 0.0466±0.0000 | 0.2340±0.0000 |
| RF | 0.0402±0.0007 | 0.2373±0.0012 |
| TEXWPT-v4-lite(no physics) | 0.0446±0.0096 | 0.2524±0.0064 |
| Ridge | 0.0406±0.0000 | 0.2776±0.0000 |
| TEXWPT-v4(full, direct) | 0.0438±0.0061 | 0.4140±0.0727 |

渠道缺失记录: TabPFN import failed; TabNet(pytorch-tabnet) not installed