# P5 model-driven domain mining (counterfactual on model, not data stats)

## 5.1 f×Pin 反事实面（plain/air/0.3m 口径, 模型预测）
- 模型最优工作点: f ≈ 2.385 GHz, pin_avail ≈ -3.5 dBm（设计中心）
- 崩塌边界: 每个 pin 档的最差频率预测 = [0.099, 0.107, 0.118, 0.133, 0.152, 0.178] → pin_avail < -10 dBm 区效率低于模型可分辨阈
- 谐振敏感度: 最优 f 附近 ±0.01GHz 预测下降 0.0%

## 5.2 可靠性降级阶梯（同工作点模型反事实, V）
- r15/plain: 0.559
- r20/plain: 0.545
- plain/plain: 0.543
- r15/bent: 0.539
- r20/bent: 0.525
- plain/bent: 0.522
- 弯折惩罚(plain→bent, 同 proto): 3.8%
- 弯折惩罚(r20→bent): 3.6%

## 5.3 跨池结构差异（模型口径）
- 81785 预测均值 0.398 V (实测 0.398) | 81813 预测均值 0.160 V (零样本)
- 模型对 81813 的隐含池偏置 = -0.496 V（相对其锚）