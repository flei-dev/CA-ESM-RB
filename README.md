# CA-ESM-RB: Context-Aware Energy-Efficient Perceptual Bitrate Adaptation

Code and data for the manuscript *"Context-Aware Energy-Efficient Perceptual Bitrate Adaptation for Mobile Streaming Services"*.

CA-ESM-RB extends the energy-aware perceptual ABR policy ESM-RB with (i) recent throughput statistics and (ii) manifest-provided next-chunk VMAF and size for every representation, and trains with random content starts. Evaluation uses leave-one-video-out policy fitting, trace-driven simulation, and measured playback-energy tables from two phones.

## Repository layout

```
training/
  esm_tmm/              simulator, learner, baselines, experiment driver, analysis (pure numpy/pandas/scipy)
  cooked_traces/        213 training network traces
evaluation/
  test_sim_traces/      92 test network traces (disjoint from training)
  evaluationFiles/      per-chunk sizes and phone-model VMAF of the three videos (tos, bbb, doc)
  powerMeasurementFiles/ measured per-chunk playback energy, Galaxy S4 and Galaxy XCover Pro
  rep_6/test_results/   replay logs of rule-based methods and NetLLM, rescored by esm_tmm.analyze
netllm/                 script that evaluates the official NetLLM (Llama2-7B) ABR checkpoint in this simulator
scripts/                figure scripts for the paper
results/                aggregated results used in the paper (see "Results" below)
```

## Setup

Python 3.10, CPU only:

```bash
pip install -r requirements.txt
cd training
python -m esm_tmm.selftest          # about one minute
```

## Reproducing the experiments

All commands are run from `training/`.

```bash
# list / run the experiment matrix (each run trains one policy; 30,000 episodes)
python -m esm_tmm.experiments list
python -m esm_tmm.experiments run --worker 0 --workers 1 --jobs 16
python -m esm_tmm.experiments status

# energy-weight selection on training-side validation sessions
python -m esm_tmm.select_weight

# aggregate all runs and baselines into tables and statistical tests
python -m esm_tmm.analyze
```

A single run, e.g. CA-ESM-RB with *tos* held out:

```bash
python -m esm_tmm.run --config '{"variant":"ctx","test_video":"tos","seed":0}' --out ../results/esm_tmm/runs_v2
```

Method variants (`esm_tmm/config.py`): `ctx` = CA-ESM-RB (proposed), `esmrb` = ESM-RB, `ctx_nolook` / `look_only` = state ablations, `greenabr` / `pensieve` / `genet` = learned baselines re-implemented and retrained under the same protocol (`esm_tmm/baselines.py`). The `cons` / `full*` variants are exploratory energy-budget versions and are not used in the paper.

NetLLM: see `netllm/run_netllm.py` (requires a GPU, the NetLLM repository and its released ABR checkpoint; Llama2-7B weights are not redistributed here).

## Protocol summary

| Item | Setting |
|---|---|
| Videos | Tears of Steel, Big Buck Bunny, a documentary; 45 chunks of 4 s; 300/750/1200/1850/2850/4300 kbps |
| Folds | leave-one-video-out (3 folds); test traces never used for training or selection |
| Seeds | 5 per learned method (NetLLM: 3 sampling seeds, official weights) |
| QoE | SQoE-III linear model |
| Energy | measured playback energy + network energy + rebuffering baseline power; efficiency = 1000·QoE/E, per session and device |
| Statistics | paired Wilcoxon signed-rank tests over 276 (video, trace) sessions |

## Results

`results/report.md` and the CSV files contain all aggregated numbers. The paper uses section A2 (main comparison with validation-selected energy weights) and sections B–F (ablation, energy-weight sweep, cross-device, power-model transfer, fixed vs. random starts). Rows marked with `*` other than NetLLM are author-released checkpoints of earlier methods; they may have seen the test videos during training and are **not** used in the paper.

## Data sources

The network traces (HSDPA 3G, Belgium 4G/LTE, FCC, synthetic), the video size/VMAF files and the phone power measurements come from the public GreenABR / ESM-RB evaluation setup and the trace collections they build on. They are included only so that the experiments can be reproduced; please cite and follow the licences of the original sources:

- B. O. Turkkan et al., "GreenABR: Energy-Aware Adaptive Bitrate Streaming with Deep Reinforcement Learning," ACM MMSys 2022.
- H. Mao, R. Netravali, M. Alizadeh, "Neural Adaptive Video Streaming with Pensieve," ACM SIGCOMM 2017.

## Citation

If you use this code, please cite the paper (citation will be added after publication) and the conference version:

W. Ke, Y. Wang, S. Shen, Y. Wang, Z. Chen, Y. Liang, J. Chen, and Y. Zhang, "ESM-RB: Energy-Aware Perceptual Bitrate Adaptation for Mobile Web Streaming Services," IEEE ICWS 2026, doi: 10.1109/ICWS72778.2026.00088.

## License

Code: MIT (see `LICENSE`). Data files remain under the terms of their original sources.
