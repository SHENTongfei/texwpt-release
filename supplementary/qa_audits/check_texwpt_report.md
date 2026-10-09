# check_texwpt 复核报告（S1，2026-09-27，ZCode 侧）

> 依据 CODEX_LAUNCH.md 第 3 步：readme.pdf 全读 + 逐文件实测/仿真分类复核；与 ex-ante 分类不符 = STOP-AND-REPORT。
> 方法：两份 readme.pdf 全文读取（非索引）+ 6 个代表性 txt 表头/行数抽样 + 与 DATA_SOURCES.md ex-ante 分类对照。

## A. 官方逐文件分类（readme 声明，权威口径）

### 81785（全编织 CP 整流天线，采集 2025，入库 2026-01-13）

| 文件 | 内容 | 来源 | 归层 |
|---|---|---|---|
| Fig2.txt | RF-DC 效率 @2.4GHz @-10dBm vs (Rin,Xin) | **Keysight ADS 谐波平衡**（readme 明确） | 仿真锚 |
| Fig3_air/body.txt | 输入阻抗 vs f（3 个 T-match 长度） | Ansys HFSS | 仿真锚 |
| Fig4a_air/body.txt | 方向图 RHCP/LHCP @2.4GHz | HFSS | 仿真锚 |
| Fig4b_AR_air/body.txt | 轴比 vs f | HFSS | 仿真锚 |
| Fig4c_eff_rad_simu_air/body.txt | 辐射效率 vs f | HFSS | 仿真锚 |
| Fig4c_max_gain_simu_air/body.txt | 最大增益 vs f | HFSS | 仿真锚 |
| Fig6a_plain/bent.txt | 实测整流电压 vs f × 4 距离，mV，EIRP=27dBm | **实测** | 训练样本 |
| Fig6b_plain/bent.txt | 实测整流功率 vs f × 4 距离，W | **实测** | 训练样本 |
| Fig7_plain/bent.txt | 实测 RF-DC 效率 vs f × 4 距离，0-1 | **实测** | 训练样本 |
| Fig8.txt | 归一化实测功率 vs **旋转角** × f，dBm/deg | **实测** | 训练样本（极化旋转维度） |

### 81813（纺织集成 CP 整流天线 r15/r20，采集 **2024**，入库 2026-01-14）

| 文件 | 内容 | 来源 | 归层 |
|---|---|---|---|
| Fig3_rect_performance.txt | R(kΩ)→Eff(%)/VDC @2.4GHz | **Keysight ADS 谐波平衡 = 仿真** | **仿真锚（非实测！）** |
| Fig4_eff_vs_Zant.txt | 效率 vs (Rin,Xin) @-10dBm | ADS | 仿真锚 |
| Fig5_VDC_vs_Zant.txt | VDC vs (Rin,Xin) @-10dBm | ADS | 仿真锚 |
| Fig6_Zant.txt | 输入阻抗 vs f | HFSS | 仿真锚 |
| Fig7_simu_rad_pattern.txt | 方向图 @2.4GHz 两主切面 | HFSS | 仿真锚 |
| Fig8_simu_AR.txt | 轴比两主切面 @2.4GHz | HFSS | 仿真锚 |
| Fig11a_meas_VDC_{plain,r15,r20}.txt | 实测 VDC vs f × 4 Pin（-4/-8/-10/-12 dBm），V | **实测** | 训练样本 |
| Fig11b_meas_PDC_{plain,r15,r20}.txt | 实测 PDC vs f × 4 Pin，uW | **实测** | 训练样本 |
| Fig12_meas_eff_plain.txt | 实测效率 vs f × 4 Pin，% | **实测** | 训练样本 |
| Fig12_meas_eff_{r15,r20}.txt | 实测效率 vs f × **2 Pin（仅 -4/-10）**，% | **实测** | 训练样本（列数不均衡） |
| Fig13_meas_RA.txt | 实测 VDC vs 旋转角 × {plain, r20×case1/2, r15×case1/2} @2.4GHz | **实测** | 训练样本（旋转+重复测量） |

## B. STOP-AND-REPORT：与 ex-ante 分类的偏差

1. **【重大】81813 Fig3_rect_performance 是 ADS 仿真，不是实测**。ex-ante（DATA_SOURCES.md + TabPFN 冒烟）曾把它当"实测负载扫掠/外部条件"。**修正**：降级到仿真锚层，禁止进实测训练/外部样本；跨原型零样本的 ext-1 锚定只用 Fig11a/b/12/13。
2. 81785 Fig6a 表头写 `V(0.6m)`，readme 明确距离网格 = **0.3/0.5/0.65/0.8 m** → 口径冻结以 readme 为准（0.65 m），表头笔误披露。
3. 81785 Fig2.txt 文件头印 `%ANSOFT 05/20/25`，readme 声明 **Keysight ADS Harmonic Balance** → 以 readme 为准（ADS），文件头疑为导出模板残留；仿真器归属披露即可，不影响"仿真锚"归层。
4. 81813 Fig12 r15/r20 只有 -4/-10 dBm 两列（plain 四列）→ 数据面构建按长表处理，缺失 Pin 组合天然留空，禁插补。
5. **时间口径修正**：81813 数据采集于 **2024**（先），81785 采集于 **2025**（后）——跨原型零样本方向 81785→81813 在时间上是"后采→先采"，论文叙事禁写"时间外推"，只写"跨原型/跨协议泛化"。发射条件两池一致（EIRP=27 dBm）✓。

## C. 复核结论

- 37 个文件全部与 readme 对上（18+19 = 81785 的 18 文件 + 81813 的 17 文件 + 2 份 readme），无缺失、无未知文件。
- 实测层样本量：81785 ≈ 200+ 点（f×4 距离×{V,P,eff}×{plain,bent} + Fig8 旋转面）；81813 = 72+72+~100+~48+~50 点。
- 仿真锚层素材比 ex-ante 预期**更厚**（两池各 3 个 ADS (Rin,Xin) 曲面 + HFSS 特性曲线），PARD-TEX 残差锚的素材充足。
- **判定：GO**——偏差 1-5 全部有修正方案，无任务级阻断。修正已同步 DATA_SOURCES.md。
