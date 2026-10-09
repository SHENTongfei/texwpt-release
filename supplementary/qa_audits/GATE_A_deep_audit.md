# Gate A 深度审计报告（极度敌对视角，2026-09-27）

> 立场：假设作者在不注意处作弊。方法：全链构造链逐行追杀 + 统计探针实证 + 攻击自身显著性流程。

## A1 特征构造链逐行追杀（features.py 全文）

| 特征 | 构造链 | 标签接触 | 判定 |
|---|---|---|---|
| pin_avail(81785) | 27(EIRP,readme) + Gr(HFSS) + PG(f,d 解析式) | 无 | PASS |
| pin_avail(81813) | readme 输入口径 Pin_dBm | 无（输入条件非结果） | PASS |
| anchor_V(81785) | √(Pin·eff_ADS(Z_HFSS(f))·R_EST) | 无（全仿真+解析） | PASS（R_EST 披露见 A3） |
| anchor_V(81813) | ADS Fig5 曲面(Zant(f)) × 10^(ΔdB/20) | 无 | PASS |
| df_res_GHz | f − f_res；f_res = HFSS \|Z(f)\| 带内谷（2.32/2.30） | 无（整改后） | PASS |
| Q_proxy | ADS 效率轨迹峰宽（仿真） | 无（整改后） | PASS（81813 退化为池常数=P2 冗余，消融检验） |
| smith_ang/norm | HFSS Z(f) 轨迹 + 曲线自身 max 归一 | 无（整改后） | PASS |
| sq/sat_shape | pin_avail 解析函数 | 无 | PASS |

## A2 统计探针（Spearman vs Y，实证而非目测）

- 全 df：无一特征 |ρ|>0.9（最高 pin 系 +0.79 = 物理驱动，合法）；修复后 df_res=−0.06 / Q=−0.25 → 泄漏信号消失 ✓
- 81813 测试行 anchor_V ρ=+0.955：**敌对拷问 = 非泄漏**——锚构建零标签接触，强相关来自 ADS 仿真与同池实测的真实一致性（此即"仿真锚强"领域发现的实证）；仿真文件为公开出版数据（CC-BY-NC-ND），非实测标签
- Q_proxy(81813) = 常数（ConstantInput 警告）→ 无信息无害

## A3 跨池信息流（敌对方向性检查）

- R_EST 由 81813 ADS Fig3rect 定标 → 进入 81785 锚标度。方向：**测试池仿真参数 → 训练特征**。敌对追问"是否给 EX 开后门"：81813 预测路径（自己的 ADS 曲面锚 + is81=0 不经 I9 修正）不经过 R_EST 修正项 → 无直接后门。判定 = **合法域知识 + ex-ante 披露义务**（论文 Methods 声明）
- I9 logR 训练学得值在 81813 行不作用（is81_te=0）✓ 代码实证

## A4 攻击自身显著性流程

- SEEDS=[42,2024,7,123,99,555,777,31337] 一次性先验定死，无中途增删 ✓
- 8 seeds 全报（无隐藏坏 seed——verdict json 全量在案）✓
- Wilcoxon/Holm/bootstrap 三口径并报，含 ns 和边缘结果全部展示 ✓
- 遗留：I7 门控架构在 81813 上发明 = **模型选择偏差（P0-3 维持）**→ 补救 = MHN 真盲外部终确认（S3 执行），论文披露

## A5 残留雷处置

- build_dataset.py 旧泄漏版 f_res → 列改名 `f_res_LEAKY_DO_NOT_USE` 隔离 + df_res 移交 features.py ✓（CSV 已重建）
- 判定：任何未来脚本 import features 或读 CSV 均不再接触泄漏列

## Gate A 裁决：**PASS（条件件）**

- 条件件：A4 的 MHN 盲确认（S3）、R_EST 披露写入论文 Methods（S7）
- 下一门：S1 基线扩军 → Gate 1 审计
