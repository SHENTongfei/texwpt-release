# S6 model-driven mining (TEXWPT-I7 ensemble counterfactuals)

## 6.1 可靠性降级阶梯（同工作点, 耦合×原型×机械, V）
- body/r15/plain: 0.453
- air/r15/plain: 0.453
- body/r15/bent: 0.451
- air/r15/bent: 0.448
- body/plain/plain: 0.443
- air/plain/plain: 0.443
- air/r20/plain: 0.442
- body/r20/plain: 0.441
- body/plain/bent: 0.439
- air/plain/bent: 0.436
- body/r20/bent: 0.431
- air/r20/bent: 0.429
- 体载惩罚(plain/plain: air→body): 0.0% (基点=自身)
- 体载代价(air→body, plain/plain): -0.1%
- 体载代价(r15/bent vs air/plain): -1.8% (最恶劣组合总降级)
- 弯折代价(air, plain→bent): 1.5%

## 6.2 f×pin 反事实面（plain/air/0.3m, 3-seed 集成）
- 设计中心: f≈2.360 GHz, pin_avail≈-12.0 dBm (max V=0.566)
- 每 pin 档最差频率预测: [0.417, 0.417, 0.417, 0.417, 0.417] → pin≤-12dBm 死区 V<0.11
- f 敏感度: 峰值频率 ±0.01GHz 内平均下降 11.4%
- 门控均值(81813 全测试行): 0.497 V vs 实测 0.532 V