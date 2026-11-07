# CA-ESM-RB experiment report

## A. Overall comparison (held-out video, mean ± std over 5 seeds)

Learned methods marked * were trained by their authors (content may overlap the test video); ours and ESM-RB never see the test video.

| method | Efficiency_Avg | QoE | Energy_Avg | Rebuffer_Time | Smoothness_Change | VMAF_mean | Data |
|---|---|---|---|---|---|---|---|
| ESM-RB (ICWS ckpt)* | 6.554 | 141.474 | 23814.338 | 1.663 | 11.576 | 55.224 | 26369172.859 |
| GreenABR* | 6.270 | 142.882 | 24670.004 | 1.659 | 12.942 | 56.849 | 30876477.967 |
| CA-ESM-RB (ours) | 6.031 ± 0.111 | 144.535 ± 1.563 | 27811.117 ± 1162.440 | 3.385 ± 0.687 | 13.539 ± 0.509 | 58.356 ± 0.656 | 32826129.612 ± 1649513.373 |
| ESM-RB | 5.829 ± 0.113 | 132.126 ± 2.020 | 25333.048 ± 729.182 | 2.058 ± 0.395 | 14.190 ± 0.568 | 54.513 ± 0.759 | 30741717.358 ± 2043449.818 |
| Genet (retrained) | 5.410 ± 0.130 | 120.537 ± 4.054 | 24504.442 ± 1040.156 | 1.660 ± 0.264 | 14.828 ± 0.238 | 50.989 ± 1.350 | 28221129.017 ± 3786240.990 |
| GreenABR (retrained) | 5.344 ± 0.136 | 133.101 ± 2.764 | 28568.910 ± 953.983 | 2.886 ± 0.596 | 15.201 ± 0.251 | 55.634 ± 1.155 | 39628380.364 ± 1431949.504 |
| Pensieve (retrained) | 5.310 ± 0.204 | 117.277 ± 3.109 | 24076.091 ± 661.487 | 1.464 ± 0.110 | 14.142 ± 0.400 | 49.384 ± 1.068 | 27055154.495 ± 2577924.700 |
| MPC | 5.293 | 146.931 | 31621.371 | 4.000 | 16.493 | 61.856 | 46224077.920 |
| Pensieve* | 5.183 | 143.800 | 33408.793 | 5.888 | 15.149 | 59.906 | 46864359.518 |
| NetLLM* | 5.075 | 117.490 | 25509.901 | 1.667 | 13.225 | 48.889 | 31465791.781 |
| BOLA-E | 5.056 | 140.307 | 31038.999 | 4.119 | 13.761 | 57.907 | 42001075.431 |
| DynamicDash | 5.044 | 140.829 | 31288.988 | 4.261 | 13.862 | 58.205 | 42296536.761 |
| Dynamic ABR | 4.779 | 139.544 | 33548.502 | 5.361 | 15.431 | 59.488 | 45220675.214 |
| BOLA | 4.550 | 135.492 | 34312.144 | 5.652 | 16.449 | 59.363 | 45197157.518 |
| Throughput | 4.463 | 112.572 | 27835.551 | 3.384 | 13.540 | 49.292 | 31655126.717 |
| Genet* | 4.236 | 129.987 | 39192.679 | 8.835 | 14.830 | 58.001 | 46723917.482 |


### Per held-out video

| method | test_video | QoE | Energy_Avg | Efficiency_Avg |
|---|---|---|---|---|
| GreenABR* | bbb | 118.4 | 2.611e+04 | 4.973 |
| ESM-RB (ICWS ckpt)* | bbb | 113.9 | 2.569e+04 | 4.949 |
| CA-ESM-RB (ours) | bbb | 121.9 | 3.098e+04 | 4.773 |
| NetLLM* | bbb | 111.3 | 2.651e+04 | 4.64 |
| Pensieve (retrained) | bbb | 104.4 | 2.525e+04 | 4.499 |
| Genet (retrained) | bbb | 102.2 | 2.568e+04 | 4.356 |
| MPC | bbb | 125.1 | 3.273e+04 | 4.327 |
| Pensieve* | bbb | 122.7 | 3.509e+04 | 4.214 |
| BOLA-E | bbb | 116.2 | 3.298e+04 | 3.921 |
| DynamicDash | bbb | 116.6 | 3.358e+04 | 3.896 |
| ESM-RB | bbb | 91.27 | 2.711e+04 | 3.881 |
| GreenABR (retrained) | bbb | 103 | 3.042e+04 | 3.8 |
| Dynamic ABR | bbb | 116.6 | 3.546e+04 | 3.755 |
| BOLA | bbb | 116.3 | 3.551e+04 | 3.749 |
| Genet* | bbb | 113.3 | 4.039e+04 | 3.645 |
| Throughput | bbb | 89.4 | 2.965e+04 | 3.295 |
| ESM-RB (ICWS ckpt)* | doc | 88.27 | 2.252e+04 | 4.293 |
| GreenABR* | doc | 86.77 | 2.283e+04 | 4.154 |
| ESM-RB | doc | 86.57 | 2.42e+04 | 3.91 |
| CA-ESM-RB (ours) | doc | 78.17 | 2.574e+04 | 3.407 |
| GreenABR (retrained) | doc | 81.07 | 2.941e+04 | 3.304 |
| Genet (retrained) | doc | 69.71 | 2.425e+04 | 3.136 |
| Pensieve (retrained) | doc | 69.68 | 2.381e+04 | 3.128 |
| MPC | doc | 84.28 | 3.064e+04 | 3.1 |
| Pensieve* | doc | 79.93 | 3.24e+04 | 3.018 |
| DynamicDash | doc | 79.43 | 3.037e+04 | 2.969 |
| Throughput | doc | 70.86 | 2.719e+04 | 2.956 |
| BOLA-E | doc | 78.52 | 3.014e+04 | 2.95 |
| Dynamic ABR | doc | 78.86 | 3.261e+04 | 2.836 |
| BOLA | doc | 75.61 | 3.421e+04 | 2.542 |
| NetLLM* | doc | 58.6 | 2.577e+04 | 2.387 |
| Genet* | doc | 69.74 | 3.874e+04 | 2.207 |
| ESM-RB (ICWS ckpt)* | tos | 222.2 | 2.324e+04 | 10.42 |
| CA-ESM-RB (ours) | tos | 233.6 | 2.672e+04 | 9.912 |
| ESM-RB | tos | 218.5 | 2.469e+04 | 9.697 |
| GreenABR* | tos | 223.5 | 2.507e+04 | 9.682 |
| GreenABR (retrained) | tos | 215.2 | 2.588e+04 | 8.926 |
| Genet (retrained) | tos | 189.7 | 2.358e+04 | 8.738 |
| MPC | tos | 231.4 | 3.149e+04 | 8.451 |
| Pensieve* | tos | 228.8 | 3.273e+04 | 8.316 |
| Pensieve (retrained) | tos | 177.7 | 2.317e+04 | 8.302 |
| BOLA-E | tos | 226.2 | 3e+04 | 8.298 |
| DynamicDash | tos | 226.4 | 2.991e+04 | 8.268 |
| NetLLM* | tos | 182.6 | 2.424e+04 | 8.199 |
| Dynamic ABR | tos | 223.2 | 3.257e+04 | 7.746 |
| BOLA | tos | 214.5 | 3.322e+04 | 7.359 |
| Throughput | tos | 177.5 | 2.667e+04 | 7.138 |
| Genet* | tos | 207 | 3.845e+04 | 6.857 |




### Paired Wilcoxon tests (ours vs …, over video × trace, seed-averaged)

| vs | metric | median_diff(ours-other) | p_value | n_pairs |
|---|---|---|---|---|
| ESM-RB | Efficiency_Avg | 0.2593 | 6.077e-05 | 276 |
| ESM-RB | QoE | 13.12 | 2.139e-16 | 276 |
| ESM-RB | Energy_Avg | 678.1 | 5.215e-25 | 276 |
| ESM-RB | Rebuffer_Time | 0 | 1.748e-10 | 276 |
| GreenABR (retrained) | Efficiency_Avg | 0.8292 | 1.75e-24 | 276 |
| GreenABR (retrained) | QoE | 7.551 | 1.979e-17 | 276 |
| GreenABR (retrained) | Energy_Avg | -1258 | 1.31e-08 | 276 |
| GreenABR (retrained) | Rebuffer_Time | 0 | 0.0006287 | 276 |
| Pensieve (retrained) | Efficiency_Avg | 0.5613 | 7.629e-18 | 276 |
| Pensieve (retrained) | QoE | 20.12 | 4.417e-41 | 276 |
| Pensieve (retrained) | Energy_Avg | 1613 | 1.121e-30 | 276 |
| Pensieve (retrained) | Rebuffer_Time | 0 | 2.368e-19 | 276 |
| Genet (retrained) | Efficiency_Avg | 0.4232 | 4.632e-15 | 276 |
| Genet (retrained) | QoE | 18.37 | 6.648e-39 | 276 |
| Genet (retrained) | Energy_Avg | 1471 | 1.073e-26 | 276 |
| Genet (retrained) | Rebuffer_Time | 0 | 9.862e-17 | 276 |
| BOLA | Efficiency_Avg | 1.204 | 1.05e-42 | 276 |
| BOLA | QoE | 1.444 | 6.923e-05 | 276 |
| BOLA | Energy_Avg | -5676 | 5.168e-37 | 276 |
| BOLA | Rebuffer_Time | -0.3785 | 3.041e-28 | 276 |
| BOLA-E | Efficiency_Avg | 0.7395 | 2.051e-28 | 276 |
| BOLA-E | QoE | -3.155 | 0.5327 | 276 |
| BOLA-E | Energy_Avg | -3083 | 8.094e-18 | 276 |
| BOLA-E | Rebuffer_Time | -0.2132 | 1.829e-12 | 276 |
| MPC | Efficiency_Avg | 0.4456 | 3.35e-22 | 276 |
| MPC | QoE | -8.658 | 1.669e-08 | 276 |
| MPC | Energy_Avg | -3841 | 1.206e-31 | 276 |
| MPC | Rebuffer_Time | -0.0001122 | 0.6648 | 276 |
| DynamicDash | Efficiency_Avg | 0.7237 | 1.927e-29 | 276 |
| DynamicDash | QoE | -3.421 | 0.91 | 276 |
| DynamicDash | Energy_Avg | -3273 | 5.916e-19 | 276 |
| DynamicDash | Rebuffer_Time | -0.2214 | 3.678e-13 | 276 |
| Dynamic ABR | Efficiency_Avg | 0.8988 | 1.239e-36 | 276 |
| Dynamic ABR | QoE | -0.703 | 0.1656 | 276 |
| Dynamic ABR | Energy_Avg | -4170 | 8.396e-31 | 276 |
| Dynamic ABR | Rebuffer_Time | -0.2539 | 9.29e-22 | 276 |
| Genet* | Efficiency_Avg | 1.419 | 3.255e-41 | 276 |
| Genet* | QoE | 6.757 | 3.214e-14 | 276 |
| Genet* | Energy_Avg | -6080 | 3.051e-32 | 276 |
| Genet* | Rebuffer_Time | -0.2928 | 9.227e-14 | 276 |
| ESM-RB (ICWS ckpt)* | Efficiency_Avg | -0.3428 | 1.485e-25 | 276 |
| ESM-RB (ICWS ckpt)* | QoE | 1.226 | 0.003972 | 276 |
| ESM-RB (ICWS ckpt)* | Energy_Avg | 2243 | 2.774e-44 | 276 |
| ESM-RB (ICWS ckpt)* | Rebuffer_Time | 0.003411 | 1.313e-17 | 276 |
| GreenABR* | Efficiency_Avg | -0.1698 | 8.106e-05 | 276 |
| GreenABR* | QoE | 1.192 | 0.07416 | 276 |
| GreenABR* | Energy_Avg | 1005 | 1.687e-25 | 276 |
| GreenABR* | Rebuffer_Time | 0 | 8.021e-15 | 276 |
| NetLLM* | Efficiency_Avg | 0.8968 | 1.582e-24 | 276 |
| NetLLM* | QoE | 22.92 | 2.126e-38 | 276 |
| NetLLM* | Energy_Avg | 567.9 | 1.041e-05 | 276 |
| NetLLM* | Rebuffer_Time | 6.939e-18 | 4.704e-20 | 276 |
| Pensieve* | Efficiency_Avg | 0.5336 | 3.706e-21 | 276 |
| Pensieve* | QoE | -6.271 | 4.885e-05 | 276 |
| Pensieve* | Energy_Avg | -3413 | 6.936e-17 | 276 |
| Pensieve* | Rebuffer_Time | -0.105 | 2.711e-12 | 276 |
| Throughput | Efficiency_Avg | 1.232 | 2.41e-37 | 276 |
| Throughput | QoE | 24.48 | 3.965e-27 | 276 |
| Throughput | Energy_Avg | -93.4 | 0.6302 | 276 |
| Throughput | Rebuffer_Time | -0.1789 | 1.132e-07 | 276 |




## A2. Main comparison, energy weight of each learned method chosen on its training videos (5 seeds)

Validation efficiency by weight: {"esmrb": {"0.25": 4.2856113997712, "0.5": 4.3849162998490145, "1.0": 4.319904112105796}, "ctx": {"0.25": 4.410760457308075, "0.5": 4.398430629332097, "1.0": 4.547159015505937}}

| method | Efficiency_Avg | QoE | Energy_Avg | Rebuffer_Time | Smoothness_Change | VMAF_mean | Data |
|---|---|---|---|---|---|---|---|
| ESM-RB (ICWS ckpt)* | 6.554 | 141.474 | 23814.338 | 1.663 | 11.576 | 55.224 | 26369172.859 |
| GreenABR* | 6.270 | 142.882 | 24670.004 | 1.659 | 12.942 | 56.849 | 30876477.967 |
| CA-ESM-RB (ours) | w=1.0 | 6.234 ± 0.129 | 139.042 ± 3.597 | 25749.241 ± 1345.734 | 2.831 ± 0.584 | 13.451 ± 0.633 | 56.499 ± 0.998 | 26838597.817 ± 1310217.511 |
| ESM-RB | w=0.5 | 5.829 ± 0.113 | 132.126 ± 2.020 | 25333.048 ± 729.182 | 2.058 ± 0.395 | 14.190 ± 0.568 | 54.513 ± 0.759 | 30741717.358 ± 2043449.818 |
| Genet (retrained) | 5.410 ± 0.130 | 120.537 ± 4.054 | 24504.442 ± 1040.156 | 1.660 ± 0.264 | 14.828 ± 0.238 | 50.989 ± 1.350 | 28221129.017 ± 3786240.990 |
| GreenABR (retrained) | 5.344 ± 0.136 | 133.101 ± 2.764 | 28568.910 ± 953.983 | 2.886 ± 0.596 | 15.201 ± 0.251 | 55.634 ± 1.155 | 39628380.364 ± 1431949.504 |
| Pensieve (retrained) | 5.310 ± 0.204 | 117.277 ± 3.109 | 24076.091 ± 661.487 | 1.464 ± 0.110 | 14.142 ± 0.400 | 49.384 ± 1.068 | 27055154.495 ± 2577924.700 |
| MPC | 5.293 | 146.931 | 31621.371 | 4.000 | 16.493 | 61.856 | 46224077.920 |
| Pensieve* | 5.183 | 143.800 | 33408.793 | 5.888 | 15.149 | 59.906 | 46864359.518 |
| NetLLM* | 5.075 | 117.490 | 25509.901 | 1.667 | 13.225 | 48.889 | 31465791.781 |
| BOLA-E | 5.056 | 140.307 | 31038.999 | 4.119 | 13.761 | 57.907 | 42001075.431 |
| DynamicDash | 5.044 | 140.829 | 31288.988 | 4.261 | 13.862 | 58.205 | 42296536.761 |
| Dynamic ABR | 4.779 | 139.544 | 33548.502 | 5.361 | 15.431 | 59.488 | 45220675.214 |
| BOLA | 4.550 | 135.492 | 34312.144 | 5.652 | 16.449 | 59.363 | 45197157.518 |
| Throughput | 4.463 | 112.572 | 27835.551 | 3.384 | 13.540 | 49.292 | 31655126.717 |
| Genet* | 4.236 | 129.987 | 39192.679 | 8.835 | 14.830 | 58.001 | 46723917.482 |


### A2 per held-out video

| variant | test_video | QoE | Energy_Avg | Efficiency_Avg |
|---|---|---|---|---|
| ctx | bbb | 113.8 | 2.938e+04 | 4.745 |
| ctx | doc | 78.46 | 2.31e+04 | 3.78 |
| ctx | tos | 224.9 | 2.477e+04 | 10.18 |
| esmrb | bbb | 91.27 | 2.711e+04 | 3.881 |
| esmrb | doc | 86.57 | 2.42e+04 | 3.91 |
| esmrb | tos | 218.5 | 2.469e+04 | 9.697 |
| genet | bbb | 102.2 | 2.568e+04 | 4.356 |
| genet | doc | 69.71 | 2.425e+04 | 3.136 |
| genet | tos | 189.7 | 2.358e+04 | 8.738 |
| greenabr | bbb | 103 | 3.042e+04 | 3.8 |
| greenabr | doc | 81.07 | 2.941e+04 | 3.304 |
| greenabr | tos | 215.2 | 2.588e+04 | 8.926 |
| pensieve | bbb | 104.4 | 2.525e+04 | 4.499 |
| pensieve | doc | 69.68 | 2.381e+04 | 3.128 |
| pensieve | tos | 177.7 | 2.317e+04 | 8.302 |




### A2 paired Wilcoxon tests

| vs | metric | median_diff(ours-other) | p_value |
|---|---|---|---|
| ESM-RB | Efficiency_Avg | 0.4933 | 1.805e-16 |
| ESM-RB | QoE | 5.403 | 7.389e-07 |
| ESM-RB | Energy_Avg | -771 | 2.704e-07 |
| GreenABR (retrained) | Efficiency_Avg | 0.9492 | 1.798e-31 |
| GreenABR (retrained) | QoE | 0.8551 | 0.0001203 |
| GreenABR (retrained) | Energy_Avg | -3222 | 1.84e-21 |
| Pensieve (retrained) | Efficiency_Avg | 0.7148 | 1.089e-25 |
| Pensieve (retrained) | QoE | 16.54 | 2.504e-27 |
| Pensieve (retrained) | Energy_Avg | -22.01 | 0.2905 |
| Genet (retrained) | Efficiency_Avg | 0.5965 | 1.619e-24 |
| Genet (retrained) | QoE | 12.57 | 1.7e-25 |
| Genet (retrained) | Energy_Avg | -203.7 | 0.6758 |
| BOLA | Efficiency_Avg | 1.289 | 3.574e-44 |
| BOLA | QoE | -1.045 | 0.6599 |
| BOLA | Energy_Avg | -7818 | 3.174e-40 |
| BOLA-E | Efficiency_Avg | 0.8656 | 4.861e-34 |
| BOLA-E | QoE | -6.363 | 0.02097 |
| BOLA-E | Energy_Avg | -5064 | 2.268e-28 |
| MPC | Efficiency_Avg | 0.6766 | 1.546e-26 |
| MPC | QoE | -16.04 | 1.668e-12 |
| MPC | Energy_Avg | -5551 | 1.233e-34 |
| DynamicDash | Efficiency_Avg | 0.8387 | 2.537e-35 |
| DynamicDash | QoE | -6.814 | 0.002486 |
| DynamicDash | Energy_Avg | -4854 | 6.818e-30 |
| Dynamic ABR | Efficiency_Avg | 1.058 | 4.939e-40 |
| Dynamic ABR | QoE | -5.535 | 0.06137 |
| Dynamic ABR | Energy_Avg | -6482 | 3.444e-36 |
| Genet* | Efficiency_Avg | 1.769 | 5.155e-42 |
| Genet* | QoE | 2.062 | 0.004984 |
| Genet* | Energy_Avg | -7926 | 3.743e-39 |
| ESM-RB (ICWS ckpt)* | Efficiency_Avg | -0.1731 | 5.519e-16 |
| ESM-RB (ICWS ckpt)* | QoE | -2.96 | 0.003299 |
| ESM-RB (ICWS ckpt)* | Energy_Avg | 496.5 | 4.201e-16 |
| GreenABR* | Efficiency_Avg | -0.07996 | 0.2306 |
| GreenABR* | QoE | -5.738 | 2.817e-10 |
| GreenABR* | Energy_Avg | -510.7 | 0.0001918 |
| NetLLM* | Efficiency_Avg | 1.14 | 2.667e-29 |
| NetLLM* | QoE | 20.87 | 4.365e-32 |
| NetLLM* | Energy_Avg | -759.4 | 0.0007913 |
| Pensieve* | Efficiency_Avg | 0.7166 | 2.008e-25 |
| Pensieve* | QoE | -10.36 | 2.733e-09 |
| Pensieve* | Energy_Avg | -5345 | 6.032e-26 |
| Throughput | Efficiency_Avg | 1.33 | 6.315e-44 |
| Throughput | QoE | 17.64 | 8.625e-21 |
| Throughput | Energy_Avg | -1899 | 1.888e-09 |




## B. Ablation of the context state

| method | Efficiency_Avg | QoE | Energy_Avg | Rebuffer_Time | Smoothness_Change | VMAF_mean | Data |
|---|---|---|---|---|---|---|---|
| CA-ESM-RB (ours) | 6.031 ± 0.111 | 144.535 ± 1.563 | 27811.117 ± 1162.440 | 3.385 ± 0.687 | 13.539 ± 0.509 | 58.356 ± 0.656 | 32826129.612 ± 1649513.373 |
| ours w/o lookahead | 5.886 ± 0.189 | 136.037 ± 3.832 | 26097.501 ± 553.861 | 2.489 ± 0.257 | 13.642 ± 0.573 | 55.354 ± 1.317 | 32157737.751 ± 1527188.314 |
| ours w/o throughput stats | 5.975 ± 0.102 | 136.771 ± 2.398 | 25024.946 ± 155.617 | 1.782 ± 0.186 | 14.391 ± 0.856 | 55.762 ± 0.538 | 28459633.118 ± 1320424.574 |
| ESM-RB | 5.829 ± 0.113 | 132.126 ± 2.020 | 25333.048 ± 729.182 | 2.058 ± 0.395 | 14.190 ± 0.568 | 54.513 ± 0.759 | 30741717.358 ± 2043449.818 |


## C. Energy-weight sweep (seeds 0-2; w=0.5 is the default)

| method | Efficiency_Avg | QoE | Energy_Avg | Rebuffer_Time | Smoothness_Change | VMAF_mean | Data |
|---|---|---|---|---|---|---|---|
| CA-ESM-RB (ours) | w=0.25 | 5.832 ± 0.110 | 147.183 ± 3.128 | 29174.792 ± 1342.664 | 3.740 ± 0.693 | 13.739 ± 0.100 | 59.338 ± 1.229 | 36549364.676 ± 1831376.836 |
| CA-ESM-RB (ours) | w=0.5 | 5.978 ± 0.061 | 144.814 ± 1.822 | 28110.013 ± 923.653 | 3.550 ± 0.728 | 13.604 ± 0.708 | 58.580 ± 0.425 | 33202798.658 ± 1350376.395 |
| CA-ESM-RB (ours) | w=1.0 | 6.297 ± 0.034 | 140.255 ± 4.480 | 25454.584 ± 1251.683 | 2.701 ± 0.530 | 13.017 ± 0.281 | 56.502 ± 1.305 | 26629034.833 ± 1643415.416 |
| ESM-RB | w=0.25 | 5.845 ± 0.228 | 136.204 ± 3.285 | 25965.585 ± 514.809 | 1.977 ± 0.053 | 14.565 ± 1.560 | 55.785 ± 0.315 | 34565000.582 ± 1484171.298 |
| ESM-RB | w=0.5 | 5.787 ± 0.052 | 132.586 ± 2.443 | 25833.300 ± 349.330 | 2.265 ± 0.365 | 14.500 ± 0.152 | 55.000 ± 0.502 | 31733640.855 ± 1735817.206 |
| ESM-RB | w=1.0 | 5.862 ± 0.211 | 124.379 ± 5.952 | 23978.108 ± 810.516 | 1.894 ± 0.550 | 13.950 ± 0.831 | 52.016 ± 1.099 | 25041317.150 ± 1183728.012 |


## D. Cross-device (energy of Galaxy S4 = G, XCover Pro = X)

| method | QoE | Energy_G | Energy_X | Efficiency_G | Efficiency_X |
|---|---|---|---|---|---|
| CA-ESM-RB (ours) | trained on galaxy | 144.535 ± 1.563 | 31509.975 ± 834.013 | 24112.258 ± 1554.118 | 4.795 ± 0.067 | 7.267 ± 0.175 |
| CA-ESM-RB (ours) | trained on xcover | 146.355 ± 4.287 | 31983.830 ± 1121.742 | 24986.137 ± 1893.728 | 4.821 ± 0.096 | 7.159 ± 0.073 |
| ESM-RB | trained on galaxy | 132.126 ± 2.020 | 29737.622 ± 576.707 | 20928.474 ± 939.702 | 4.578 ± 0.075 | 7.081 ± 0.156 |
| ESM-RB | trained on xcover | 135.977 ± 4.962 | 29147.737 ± 235.187 | 20111.402 ± 433.866 | 4.776 ± 0.118 | 7.409 ± 0.147 |


## F. Training protocol: fixed vs random content start

| method | Efficiency_Avg | QoE | Energy_Avg | Rebuffer_Time | Smoothness_Change | VMAF_mean | Data |
|---|---|---|---|---|---|---|---|
| CA-ESM-RB (ours) | fixed start (ICWS protocol) | 5.670 ± 0.279 | 140.502 ± 3.136 | 28430.567 ± 724.453 | 3.532 ± 0.527 | 14.924 ± 1.255 | 58.073 ± 0.285 | 32839074.454 ± 1603358.252 |
| CA-ESM-RB (ours) | random start (ours) | 6.031 ± 0.111 | 144.535 ± 1.563 | 27811.117 ± 1162.440 | 3.385 ± 0.687 | 13.539 ± 0.509 | 58.356 ± 0.656 | 32826129.612 ± 1649513.373 |
| ESM-RB | fixed start (ICWS protocol) | 5.447 ± 0.177 | 126.261 ± 5.326 | 26292.293 ± 695.482 | 2.312 ± 0.482 | 16.533 ± 1.591 | 54.604 ± 0.960 | 31784988.686 ± 1236155.923 |
| ESM-RB | random start (ours) | 5.829 ± 0.113 | 132.126 ± 2.020 | 25333.048 ± 729.182 | 2.058 ± 0.395 | 14.190 ± 0.568 | 54.513 ± 0.759 | 30741717.358 ± 2043449.818 |


## E. Power-model generalisation (within-chunk corr = how well level-to-level energy differences are predicted)

| fit_device | eval_device | role | metric | mean | std |
|---|---|---|---|---|---|
| galaxy | galaxy | held-out video | r2 | -0.3792 | 0.7205 |
| galaxy | galaxy | held-out video | within_chunk_corr | 0.8886 | 0.04864 |
| galaxy | galaxy | training video | r2 | 0.4205 | 0.1048 |
| galaxy | galaxy | training video | within_chunk_corr | 0.9116 | 0.01801 |
| galaxy | xcover | held-out video | r2 | -3.467 | 1.882 |
| galaxy | xcover | held-out video | within_chunk_corr | 0.5145 | 0.3162 |
| galaxy | xcover | training video | r2 | -4.385 | 3.04 |
| galaxy | xcover | training video | within_chunk_corr | 0.5258 | 0.2853 |
| xcover | galaxy | held-out video | r2 | -3.122 | 1.254 |
| xcover | galaxy | held-out video | within_chunk_corr | 0.6617 | 0.2756 |
| xcover | galaxy | training video | r2 | -3.307 | 0.9137 |
| xcover | galaxy | training video | within_chunk_corr | 0.742 | 0.2055 |
| xcover | xcover | held-out video | r2 | -0.3504 | 0.7099 |
| xcover | xcover | held-out video | within_chunk_corr | 0.2705 | 0.2074 |
| xcover | xcover | training video | r2 | 0.3035 | 0.2662 |
| xcover | xcover | training video | within_chunk_corr | 0.6066 | 0.2252 |
