# TEXWPT 终局 AAR 汇总（2026-09-27，门控流水线全通关）

## 门控记录
| 门 | 审计对象 | 抓到并修复 | 裁决 |
|---|---|---|---|
| Gate A | R1 泄漏全链 + 统计探针 + 显著性流程 | Q/f_res 标签泄漏(前轮)、CSV 残留列、81813 非盲 | PASS(条件: MHN+披露) |
| Gate 1 | 14 基线军团同协议 | nn2 插值缺陷(二刀)、TabNet-lite 公平性(补 official)、KAN API/泄漏、旧 f_res 隔离 | PASS |
| Gate 2 | MHN 盲外部 (R4) | 模型级零样本不可行→诚实降级为锚层+架构层双口径 | PASS |
| Gate 3 | 终配消融 | **I8/I9 在修复锚下转负贡献→主配置重定=I7（组件价值=锚质量的函数）** | PASS |
| Gate 4 | 可解释+挖掘 | 6.2 设计中心边界伪影(修正读法)+门控外推保守性入 Limitations | PASS(修正后) |

## 终局数字（全部可溯源 data/qa/*）
- **主配置 = TEXWPT-I7**（门控残差, 干净特征, 修复锚）：8-seed EX=0.1158±0.1005 全场第一（14 基线），胜率 7/8 全对照；bootstrap vs TabPFN p=0.0014 / XGB 0.0011 / CatBoost 0.0039 / SVR 0.0077 / GP 0.0124 显著；vs LightGBM p=0.038 边缘
- TI：TabPFN 0.0187 最强（诚实保留, 教授拍板口径）；I7 0.0328 次之
- 消融：-gate 0.31 / -smithQ 0.30 / -all-sim 0.30（三崩=门控+Smith 组+锚必需）；I8/I9/-physics 新锚下冗余
- MHN 真盲外部：TEXWPT-gate CV 0.0495 与 TabPFN 0.0500 并列第一 > 锚 0.079 > LightGBM 0.29
- 可解释：anchor_V 置换 MAE +0.234（3.6×）= 锚命脉定量铁证；smith_norm 归因第一
- 挖掘：形变/体载总代价 <2%（该工作点, 反直觉）；f 敏感度 11.4%/±0.01GHz；门控外推保守性=设计性质
- 文献：T1-T6 六主题（PhiSRL=方法学直接先例）

## 遗留（不阻塞交付）
1. TI 口径教授拍板（TabPFN 最强内插的叙事处理）
2. cudnn deterministic（R3 P1-3）
3. Fig8 readme 冲突遗留 supplement
4. 逐样本预测存储（更强分层 bootstrap）
