# G1 v1 report (PARD anchors, 5 seeds)
dev=cuda
seed42: TI=0.0389  EX=0.3807
seed2024: TI=0.0335  EX=0.3716
seed7: TI=0.0251  EX=0.2794
seed123: TI=0.0270  EX=0.3270
seed99: TI=0.0261  EX=0.2577

SUM  TI v1: 0.0301±0.0053  (v0: 0.022, RF 靶 0.0536)
SUM  EX v1: 0.3233±0.0487  (v0: 0.275±0.09, RF 靶 0.1989, TabPFN 0.2804, 纯锚 0.4051)
## v1 复盘（PUA L2）与 v2 方案（口径统一，实质不同）

**v1 结果**：EX=0.3233±0.0487（方差收敛 0.09→0.049 但均值劣于 v0 0.275 与 RF 0.199）；TI=0.0301±0.0053 仍胜 RF 2 倍。
**锚特征注入没解决跨域 → 根因不在特征量，在物理口径**（读 readme 原文发现）：
- 81813 Fig11 的 Pin = **整流器输入可用功率**（readme: "available power at the rectifier input"）→ 81813 的 VDC(f,Pin) 已把天线/距离归一掉，是**整流器口径**
- 81785 Fig6 的 V(f,d) 是系统口径（EIRP 27dBm 固定，天线+距离都卷在 Y 里）
- **两池 Y 的物理口径不同 = 假跨域**。桥 = Friis：81785 样本派生 pin_avail_est = 27 + Gr^HFSS(f)(Fig4c gain 在锚层) + PG(f,d) → 两池统一到整流器口径
- 另：v1 的"纯锚对照"(曲面中位数) 太弱，非合法对照；真纯锚 = Friis 链路 + HFSS gain + ADS 整流曲面的物理管线（v2 一并实现）
**v2**：统一口径训练（f 偏移, pin_avail, 类别）+ ADS 锚特征 + 真纯锚管线对照；RF 同口径重跑。验收 = EX 腿 mean±std 双胜 RF。
