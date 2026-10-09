# S3 MHN blind external probe (R4)
rows=90 conds=5
口径披露: 865.5MHz/他整流器/他实验室 -> 模型级零样本不可行(训练域2.4GHz); 本探针=物理锚对照+架构可迁移性(MHN内部CV/holdout)

| 模型 | MHN内CV MAE | 20% holdout MAE |
|---|---|---|
| TabPFN-v2 | 0.0500±0.0000 | 0.0484±0.0100 |
| TEXWPT-gate | 0.0495±0.0014 | 0.0509±0.0104 |
| PhysAnchor | 0.0790±0.0000 | 0.0810±0.0136 |
| LightGBM | 0.2931±0.0000 | 0.2915±0.0427 |
| Ridge | 0.3949±0.0000 | 0.3624±0.0661 |