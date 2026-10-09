# TEXWPT 终裁裁决（8-seed，折级配对 Wilcoxon + Holm）

| 配置 | TI mean±std | EX mean±std |
|---|---|---|
| I7 | 0.0328±0.0135 | 0.1158±0.1005 |
| Final | 0.0444±0.0173 | 0.1174±0.0974 |
| LightGBM | 0.0985±0.0354 | 0.1904±0.0000 |
| FT-Transformer | 0.1657±0.0146 | 0.2033±0.0024 |
| GP | 0.1676±0.0148 | 0.2068±0.0000 |
| SVR | 0.0269±0.0024 | 0.2255±0.0000 |
| CatBoost | 0.0543±0.0151 | 0.2325±0.0028 |
| KNN | 0.0707±0.0209 | 0.2361±0.0000 |
| XGBoost | 0.0518±0.0203 | 0.2398±0.0000 |
| TabPFN-v2 | 0.0187±0.0021 | 0.2442±0.0000 |
| RF | 0.0443±0.0120 | 0.2521±0.0018 |
| Ridge | 0.1000±0.0404 | 0.2658±0.0000 |
| TabNet(tabnet-lite) | 0.1657±0.0154 | 0.2684±0.1095 |
| TabNet(official) | 7.7109±12.7088 | 0.4145±0.2848 |
| KAN(pykan) | 0.0751±0.0338 | 0.5139±0.5087 |

## EX 腿（跨原型零样本主战轴）配对裁决
主模型（EX 最优）= **I7**
| 对比 | ΔMAE(我们-对照, 负=赢) | p (Wilcoxon) | p (Holm) | 胜/总 seeds |
|---|---|---|---|---|
| I7 vs TabPFN-v2 | -0.1284 | 0.0156 | 0.3906 ns | 7/8 |
| Final vs TabPFN-v2 | -0.1268 | 0.0234 | 0.4219 ns | 7/8 |
| I7 vs GP | -0.0910 | 0.1953 | 1.0000 ns | 7/8 |
| Final vs GP | -0.0894 | 0.1953 | 1.0000 ns | 7/8 |
| I7 vs SVR | -0.1097 | 0.0547 | 0.6016 ns | 7/8 |
| Final vs SVR | -0.1081 | 0.0781 | 0.6250 ns | 7/8 |
| I7 vs KNN | -0.1203 | 0.0391 | 0.5469 ns | 7/8 |
| Final vs KNN | -0.1187 | 0.0234 | 0.3984 ns | 7/8 |
| I7 vs Ridge | -0.1500 | 0.0156 | 0.3750 ns | 7/8 |
| Final vs Ridge | -0.1484 | 0.0156 | 0.3594 ns | 7/8 |
| I7 vs LightGBM | -0.0746 | 0.1953 | 0.9766 ns | 7/8 |
| Final vs LightGBM | -0.0730 | 0.1953 | 0.7812 ns | 7/8 |
| I7 vs CatBoost | -0.1167 | 0.0547 | 0.5469 ns | 7/8 |
| Final vs CatBoost | -0.1151 | 0.0391 | 0.5078 ns | 7/8 |
| I7 vs XGBoost | -0.1240 | 0.0234 | 0.3750 ns | 7/8 |
| Final vs XGBoost | -0.1224 | 0.0234 | 0.3516 ns | 7/8 |
| I7 vs RF | -0.1363 | 0.0156 | 0.3438 ns | 7/8 |
| Final vs RF | -0.1347 | 0.0156 | 0.3281 ns | 7/8 |
| I7 vs FT-Transformer | -0.0875 | 0.1953 | 0.5859 ns | 7/8 |
| Final vs FT-Transformer | -0.0859 | 0.1953 | 0.3906 ns | 7/8 |
| I7 vs TabNet(tabnet-lite) | -0.1526 | 0.0547 | 0.4922 ns | 7/8 |
| Final vs TabNet(tabnet-lite) | -0.1509 | 0.0391 | 0.4688 ns | 7/8 |
| I7 vs TabNet(official) | -0.2988 | 0.0156 | 0.3125 ns | 7/8 |
| Final vs TabNet(official) | -0.2971 | 0.0156 | 0.2969 ns | 7/8 |
| I7 vs KAN(pykan) | -0.3982 | 0.0078 | 0.2109 ns | 8/8 |
| Final vs KAN(pykan) | -0.3965 | 0.0078 | 0.2031 ns | 8/8 |
| Final vs I7 | +0.0016 | 1.0000 | 1.0000 ns | 4/8 |

## TI 腿（折级配对，n=40）
- I7 vs TabPFN-v2: Δ=+0.0141 p=0.0000
- Final vs TabPFN-v2: Δ=+0.0256 p=0.0000
- I7 vs GP: Δ=-0.1348 p=0.0000
- Final vs GP: Δ=-0.1232 p=0.0000
- I7 vs SVR: Δ=+0.0059 p=0.0584
- Final vs SVR: Δ=+0.0175 p=0.0000
- I7 vs KNN: Δ=-0.0379 p=0.0000
- Final vs KNN: Δ=-0.0263 p=0.0000
- I7 vs Ridge: Δ=-0.0672 p=0.0000
- Final vs Ridge: Δ=-0.0556 p=0.0000
- I7 vs LightGBM: Δ=-0.0657 p=0.0000
- Final vs LightGBM: Δ=-0.0541 p=0.0000
- I7 vs CatBoost: Δ=-0.0215 p=0.0000
- Final vs CatBoost: Δ=-0.0100 p=0.0026
- I7 vs XGBoost: Δ=-0.0190 p=0.0002
- Final vs XGBoost: Δ=-0.0074 p=0.0344
- I7 vs RF: Δ=-0.0115 p=0.0000
- Final vs RF: Δ=+0.0001 p=0.2887
- I7 vs FT-Transformer: Δ=-0.1329 p=0.0000
- Final vs FT-Transformer: Δ=-0.1213 p=0.0000
- I7 vs TabNet(tabnet-lite): Δ=-0.1329 p=0.0000
- Final vs TabNet(tabnet-lite): Δ=-0.1214 p=0.0000
- I7 vs TabNet(official): Δ=-7.6781 p=0.0000
- Final vs TabNet(official): Δ=-7.6665 p=0.0000
- I7 vs KAN(pykan): Δ=-0.0423 p=0.0000
- Final vs KAN(pykan): Δ=-0.0308 p=0.0000
## 新锚 bootstrap（B=10000，对照-我们 Δ，正=我们赢，CI 不含 0 = 显著）

- I7 vs LightGBM: Δ=+0.0746, CI=[-0.0035,+0.1214], p=0.0377, 胜率=7/8
- I7 vs TabPFN-v2: Δ=+0.1284, CI=[+0.0499,+0.1756], p=0.0014, 胜率=7/8
- I7 vs SVR: Δ=+0.1097, CI=[+0.0311,+0.1571], p=0.0077, 胜率=7/8
- I7 vs GP: Δ=+0.0910, CI=[+0.0121,+0.1379], p=0.0124, 胜率=7/8
- I7 vs CatBoost: Δ=+0.1167, CI=[+0.0373,+0.1628], p=0.0039, 胜率=7/8
- I7 vs XGBoost: Δ=+0.1240, CI=[+0.0459,+0.1712], p=0.0011, 胜率=7/8
- Final vs LightGBM: Δ=+0.0730, CI=[-0.0040,+0.1206], p=0.0374, 胜率=7/8
- Final vs TabPFN-v2: Δ=+0.1268, CI=[+0.0522,+0.1745], p=0.0015, 胜率=7/8
- Final vs SVR: Δ=+0.1081, CI=[+0.0305,+0.1556], p=0.0062, 胜率=7/8
- Final vs GP: Δ=+0.0894, CI=[+0.0125,+0.1370], p=0.0105, 胜率=7/8
- Final vs CatBoost: Δ=+0.1151, CI=[+0.0372,+0.1622], p=0.0024, 胜率=7/8
- Final vs XGBoost: Δ=+0.1224, CI=[+0.0454,+0.1700], p=0.0011, 胜率=7/8