# S3 baseline report (2026-09-27)

**Leg TI 81785 VDC_V** (n=72, GroupKFold by f): Ridge=0.1002±0.0402, RF=0.0536±0.0151, GBM=0.0581±0.0181
**Leg TI 81785 PDC_W** (n=71, GroupKFold by f): Ridge=2.5288±0.5992, RF=1.2693±0.5138, GBM=1.3030±0.5804
**Leg TI 81785 eff** (n=69, GroupKFold by f): Ridge=0.1280±0.0542, RF=0.0835±0.0098, GBM=0.0836±0.0124

**Leg EX 81785->81813 VDC zero-shot** (train n=72, test n=108):
  Ridge: MAE_V = 0.9730 V
  RF: MAE_V = 0.1989 V
  GBM: MAE_V = 0.2007 V
  PureAnchor(81813 Fig5 ADS VDC surface median): MAE_V = 0.4051 V (surf median=0.1267 V, range [0.001301,0.5501])
  TabPFN: MAE_V = 0.2804 V