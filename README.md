# Latency as a First-Class Metric

Code and results for the paper *Latency as a First-Class Metric: Evaluating Deep
Models for Real-Time Behaviour and State Detection in Automated Systems*
(A. Hassou, K. Tadist, A. El Affar), accepted at AICSET'26.

The paper proposes an evaluation protocol for detectors that run on sensor
streams. For the same detector it reports detection quality, per-window
inference latency, detection delay and false-alarm rate, and it sets thresholds
without test labels. This repository contains the benchmark that applies the
protocol to thirteen detectors on three public datasets, the result files, and
the scripts that turn those files into every table, figure and number in the
paper.

## Main results

Measured on the laptop described below; latency is the median time of one whole
decision at batch size one on one CPU thread.

- **Behaviour recognition (WISDM).** The 1D-CNN reaches 0.901 macro-F1 at
  0.54 ms per window. The bidirectional GRU takes about
  28 times as long with no gain in quality; the LSTM takes 1.9 times as long.
- **State detection (SKAB).** Calibrating the threshold on a short nominal
  warm-up of each run lowers the mean false-alarm rate from 0.83
  to 0.28 without using labels.
- **Point-adjusted F1.** Random scores reach a point-adjusted F1 of
  0.971 on SKAB, above every trained detector, and the inflation is
  largest for the detectors that alarm latest.

## Datasets and detectors

| Dataset | Task | Channels | Sampling rate | Window |
|---|---|---|---|---|
| WISDM v1.1 | behaviour recognition, 6 activities | 3 | 20 Hz | 128 samples |
| SKAB | state detection | 8 | 1 Hz | 30 samples |
| SMD | state detection, 28 entities | 38 | 1 per minute | 100 samples |

Behaviour recognition: random forest on hand-crafted features, MLP, 1D-CNN,
TCN, LSTM, bidirectional GRU, transformer encoder. State detection: PCA,
isolation forest, and dense, convolutional, LSTM and transformer autoencoders.
The architectures are defined in `rtbench/models.py`.

## Repository layout

| Path | Content |
|---|---|
| `config.py` | paths and the experimental settings of each track |
| `rtbench/` | data loading and windowing, models, metrics, latency helpers |
| `engine/` | training and evaluation for the two tracks |
| `train_<model>.py` | one entry point per detector; `train_controls.py` for the always-alarm and random-score controls |
| `collect.py`, `make_datasets_table.py` | merge the per-model results; dataset facts |
| `rev_sensitivity_skab.py` | sensitivity of the per-run policy to q and N_c |
| `rev_seeds_smd.py` | three-seed check on six SMD entities |
| `rev_latency_real.py`, `rev_env.py` | whole-decision latency on 500 distinct test windows; record of the measurement environment |
| `bench_latency.py`, `bench_pass.py`, `merge_latency.py` | forward-pass-only latency on one repeated window, kept for comparison |
| `rev_summarize.py`, `rev_apply_latency.py` | aggregate the supplementary runs; select the whole-decision latency for the tables |
| `analyze.py` | writes `paper_out/tables` and `paper_out/figs` from `results/` |
| `verify.py` | recomputes the numbers quoted in the paper from `results/` and checks them against the generated macros |
| `results/` | result files of the runs reported in the paper |
| `paper_out/` | generated tables, macros and figures |

## Reproducing the results

The scripts were run with Python 3.9 on Windows 11. Install the pinned versions
(PyTorch from its own index for the CUDA build):

```
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Download the datasets (about 530 MB) into `data/`, or set `RTBENCH_DATA` to an
existing copy:

```
python fetch_data.py
```

Then, from the repository root:

```
train_all.bat        rem trains every detector in its own process, collects the results
run_revision.bat     rem supplementary runs, whole-decision latency, tables, verification
```

`run_revision.bat` ends by running `analyze.py` and `verify.py`. To regenerate
the tables and figures from the result files already in the repository, without
training anything:

```
python analyze.py
python verify.py
```

Each detector can also be trained on its own, for example
`python train_cnn1d.py`. Training uses the GPU when `DEVICE = "cuda"` in
`config.py`; latency is always measured on the CPU.

## About the latency numbers

The reported numbers were measured on an AMD Ryzen 7 4800H laptop (8 cores,
16 GB, Windows 11): mains power, the vendor's default power scheme, one Torch
thread, normal priority, no core pinning, thermal state not controlled. Each
pass times one decision on each of 500 distinct test windows after 100 warm-up
decisions; the pass is repeated five times in separate processes and the median
is reported. `results/rev_env.json` records the environment.

Absolute latencies depend on the machine and on its state: the same protocol
gave medians about 26% apart in two sessions weeks apart on
this laptop, and up to 25% apart between passes run minutes
apart. Expect the ordering of the detectors to carry over to other hardware
more reliably than the ratios between them.

## Data sources

- WISDM v1.1: Kwapisz, Weiss and Moore, *Activity recognition using cell phone
  accelerometers*, 2011.
- SKAB: Katser and Kozitsin, *Skoltech Anomaly Benchmark*, 2020
  (https://github.com/waico/SKAB).
- SMD: Su et al., *Robust anomaly detection for multivariate time series
  through stochastic recurrent neural network*, KDD 2019
  (https://github.com/NetManAIOps/OmniAnomaly).

The datasets are not redistributed here; each keeps its own licence.
