Reference `outputs/v69-shape/F7_v6144_h640_d32_r0` (5 seeds, mean 1.0579); gain = at least 0.005 BPC better on at least 4 of 5 seeds.

| arm | seeds | params | BPC | change vs ref | se | per seed | gain seeds | worse seeds | verdict |
|---|---|---|---|---|---|---|---|---|---|
| F7_v6144_h640_d32_bs8wd05 | 5 | 99,882 | 1.0435 | -0.0144 | 0.0016 | -0.0106 -0.0135 -0.0139 -0.0205 -0.0133 | 5 | 0 | PASS |
| F7_v6144_h640_d32_bs8wd10 | 5 | 99,882 | 1.0501 | -0.0078 | 0.0017 | -0.0058 -0.0057 -0.0067 -0.0147 -0.0062 | 5 | 0 | PASS |
| F7_v6144_h640_d32_bs8wd20 | 5 | 99,882 | 1.0743 | +0.0164 | 0.003 | +0.0279 +0.0127 +0.0171 +0.0115 +0.0129 | 0 | 5 | no |
