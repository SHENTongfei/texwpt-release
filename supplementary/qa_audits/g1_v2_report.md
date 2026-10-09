# G1 v2 report (统一整流器口径 + 真纯锚 + RF 同口径)
dev=cuda

PureAnchor(81813, Friis口径ADS外推): MAE_V = 0.1242
RF(同口径) EX: MAE_V = 0.2276
RF(同口径) TI: MAE_V = 0.0495±0.0118
seed42: TI=0.0374  EX=0.2525
seed2024: TI=0.0393  EX=0.2444
seed7: TI=0.0298  EX=0.2272
seed123: TI=0.0267  EX=0.1988
seed99: TI=0.0242  EX=0.2479

SUM TI v2: 0.0315±0.0059
SUM EX v2: 0.2341±0.0196   (RF 同口径见上; v1: 0.323±0.049)
## v2 复盘 → v3 方案（真 PARD 残差结构）

**发现 1（领域级）**：真纯锚管线（HFSS Zant(f) → ADS Fig5 整流曲面 → pin 电压外推，零实测拟合）= **0.1242 V**，打赢所有学习模型（v2 net 0.234、RF 0.228、TabPFN 0.280、v1 弱锚 0.405）。"结构化仿真锚统治小样本冷门域" = 论文的领域偏置证据之一。
**发现 2**：Q4 ex-ante 预警触发（"打不赢纯仿真锚"）→ 按预案走 L5 差距建模口径，且 PARD 的正确打开方式不是"锚当特征"而是"锚当结构"：
**v3 = yhat = anchor_phys(pool) + g_θ(feat)**，残差保底 = 0（即锚本身）：
- 81813 锚 = Fig5@Zant(f) × 10^(ΔdB/20)（已有，0.124）
- 81785 锚 = sqrt(Pin_lin × eff_ADS81(Z81(f)) × R_est)，R_est = Fig3rect ADS 的 VDC²/PDC 中位数（仿真侧定标，合法）
- 模型只学残差（ADS vs 实测的系统偏差，跨池同质）→ EX 保底 0.124，任何正残差贡献 = WIN
**验收**：EX 5 seeds 全部 < 0.124 且 TI < RF 0.0495。
