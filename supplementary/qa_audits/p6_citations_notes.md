# P6 citation 检索笔记（2026-09-27，WebSearch 多源，待 P5 知识点定稿后合并为正式支撑表）

## T1 弯折/形变 → 去谐与性能退化（支撑 I2/降级阶梯）
- Analysis on Bending Performance of the Electro-Textile Antennas (IEEE 9738630, 2022): 结构形变→频率去谐/增益带宽损失, 弯曲半径 42.5mm 实验
- Cylindrical Bending of Deformable Textile Rectangular Patch Antennas (Wiley/Hindawi 2012): 柱面弯曲解析模型（贴片拉伸+基板压缩 as f(bending radius)）
- 机理: 弯折改变电长度/基板厚度/有效介电 → 谐振频移; H 面弯折 vs 其他面差异

## T2 整流器非线性三区（支撑 I6a 物理残差基）
- Power efficiency and optimum load formulas on RF (ResearchGate 2026): 920MHz 75.9% 峰值效率+最优负载公式
- RF Energy Harvesting Using High Impedance Asymmetric Diodes (Wiley/AGU Fakharian 2021): 60-67% eff @12-21dBm, 效率随输入功率变化
- A Dual-Band Rectenna Without Impedance Matching (IEICE He 2021): 效率依赖功率电平（平方律→线性→饱和）
- 2.45GHz 世界纪录 ~90% @8W (UPC Vera): 高功率饱和/压缩效应
- 物理链: Schottky 平方律区(低功率) → 线性整流区(峰) → 饱和/击穿

## T3 体载耦合去谐（支撑 coup 维度/body 态）
- Analysis on the Effects of the Human Body on Wearable Antennas (PMC, Abd Rahman 2019, cited 90): 电纺织天线+人体效应
- Wearable Directional Button Antenna for On-Body WPT (MDPI 2023, Xu et al.): AMC 阵列抗去谐
- Wearable Antennas for Cross-Body Communication (IEEE 2020, Su et al.): 高介电人体组织邻近→谐振频移
- 机理: 人体高 ε 邻近改变有效介电环境 → 去谐; HIS/AMC 缓解

## T4 机械形变→谐振频移机理（支撑 df_res 特征）
- Textile circular antenna 弯曲半径研究 (Academia.edu): H 面弯折谐振几乎不变、其他面频移
- 拉伸改变贴片物理尺寸+导体电导率 → 频移（USM EPrints 5.17→5.3GHz）
- 机织导电纺织品拉伸改变纱线接触电阻
- Dang et al. 2021 (Queensland, cited 78+): 频率可重构 wearable textile antenna（补偿形变去谐）

## T5 Smith 轨迹不变量（支撑 smith 特征，S5 归因第一 11.5%）
- Antenna Impedance Matching Using Deep Learning (PMC8541658, Kim 2021, cited 38): 阻抗/反射数据作 ML 特征
- Adaptive Antenna Impedance Matching (IEEE 10185565, Hasan 2023, cited 27): 宽频自动匹配
- Smith chart 教材级基础 (Antenna-Theory.com / Microwaves101 / R&S): 阻抗-频率轨迹的标准诊断表达
- 定位: 轨迹形状作跨域不变量特征在纺织 WPT 是新用法（检索无先例 = 蓝海佐证）

## T6 仿真辅助残差 DL（支撑 PARD-TEX 门控残差的方法学谱系）
- **Physics-Informed Supervised Residual Learning for EM Modeling (arXiv:2104.13231, Shan 2021, cited 76)**: PhiSRL = 物理先验+残差 DL 做 EM 建模——PARD-TEX 的直接方法学先例（EM 域）→ 论文 Related Work 必引
- Physics-informed residual learning w/ spatiotemporal constraints (Sci Rep 2025, Zhu)
- ResPINNs (ICML 2026): 残差流式 PINN
