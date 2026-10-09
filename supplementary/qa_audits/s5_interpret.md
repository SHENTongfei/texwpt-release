# S5 interpretability on TEXWPT-I7 (3-seed ensemble, leak-free)

## 5.1 梯度归因（占前向总敏感度 %）
- smith_norm: 11.5%
- f_GHz: 10.2%
- anchor_V: 7.7%
- cat_2: 7.0%
- anchor_dB: 6.4%
- cat_1: 6.4%
- pin_avail: 6.1%
- sat_shape: 5.1%
- smith_ang: 4.7%
- cat_4: 4.5%
- df_res_GHz: 4.4%
- Q_proxy: 4.3%
- cat_0: 4.0%
- sq_shape: 4.0%
- cat_6: 3.5%
- cat_5: 3.5%
- cat_3: 3.4%
- pin_off_db: 3.3%

## 5.2 permutation importance（EX MAE 退化, base=0.0654）
- f_GHz: +-0.0006
- pin_avail: +-0.0033
- pin_off_db: +-0.0024
- df_res_GHz: +-0.0024
- anchor_V: +0.2337
- anchor_dB: +-0.0023
- sq_shape: +-0.0038
- sat_shape: +-0.0035
- smith_ang: +-0.0016
- smith_norm: +0.0043
- Q_proxy: +0.0000