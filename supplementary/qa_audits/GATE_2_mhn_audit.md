# Gate 2 深度审计（MHN 盲外部，极恶意视角）

## 结果（s3_mhn_probe.md）
| 模型 | MHN 内 CV | 20% holdout |
|---|---|---|
| TEXWPT-gate | **0.0495±0.0014** | 0.0509 |
| TabPFN-v2 | 0.0500 | **0.0484** |
| PhysAnchor | 0.0790 | 0.0810 |
| LightGBM | 0.2931 | 0.2915 |
| Ridge | 0.3949 | 0.3624 |

## 敌对检查链
| # | 检查 | 结果 |
|---|---|---|
| G2.1 | c 标定只用 fold-train 标签 | ✓ PASS（anchor calibration 标准做法） |
| G2.2 | GroupKFold(by dist) 分组无泄漏 | ✓ 90 点距离各异 |
| G2.3 | Type-I 红牌（公式可算？）| ✓ V=√(P·R)·eff(P)，eff(P) 非线性未闭式 → 合法；且探针目的=架构迁移非 SOTA |
| G2.4 | MHN 是否曾被接触 | ✓ 真盲（本次首评，全程无调参/选模接触） |
| G2.5 | LightGBM 崩(0.29)是否故意弱化 | ✓ 已给小样本合理配置(num_leaves=8/300 trees)仍崩 → n=90 boosting 过拟合，如实报 |
| G2.6 | TEXWPT vs TabPFN 打平的诚实呈现 | ✓ CV 我们第一/holdout 对方第一，差 0.0025 在方差内 → 论文写"并列第一/可迁移"不写"赢" |

## R4 双外部终判
- ext-1 = 81813（模型级、跨原型；**披露为 developmental external**——I7 架构在其上发明）
- ext-2 = MHN（**真盲**；物理锚层 + 架构可迁移层；模型级零样本不可行的原因已论证=训练域 2.4GHz）
- 两外部独立：不同实验室（Oviedo vs jaancis）、不同频段（2.4G vs 865.5M）、不同整流器、不同天线族 ✓
- **裁决：R4 PASS（附披露义务：81813 developmental + MHN 架构级口径写入论文）**

## 裁决：Gate 2 PASS → 放行 S4
