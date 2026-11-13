# VENMA-VAD

Code and frozen text inputs for the VENMA video anomaly detector. The repository contains the scoring
pipeline, final prompt descriptions, cached text embeddings, annotations, feature lists, the ablation
program, and the expected metrics reported in the paper.

**The per-video feature arrays are not included in this repository.** They amount to roughly 1.6 GB
(290 UCF-Crime arrays and 800 XD-Violence arrays), so they are distributed separately. Extract them with
`feature-extraction/extract_video.py` (see "Rebuild video features") or place your own copies under
`video-features/ucf/` and `video-features/xd/` before scoring.

## Contents

```text
VENMA-VAD/
├── feature-extraction/
│   ├── extract_video.py            Qwen3-VL window embeddings (creates video-features/)
│   └── encode_text.py              prompt text embeddings
├── algorithm/
│   ├── covariance.py               C
│   ├── visual.py                   V
│   ├── prompt.py                   T
│   ├── evidence.py                 E
│   ├── pipeline.py
│   ├── data_io.py
│   ├── evaluation.py
│   ├── settings.py
│   └── main.py
├── ablation/ablation.py
├── data/                           prompts, text banks, config, lists, annotations
├── results/expected_metrics.json
├── run.sh
├── requirements.txt
├── SHA256SUMS                      hashes of the files shipped here
└── SHA256SUMS.features             hashes of the external feature arrays
```

`video-features/` is created locally by the extractor and is ignored by Git.

## Required feature arrays

The scorer expects float32 Qwen3-VL-Embedding-2B window embeddings with 2,048 dimensions:

| Dataset | Arrays | Windows | Location |
|---|---:|---:|---|
| UCF-Crime | 290 | 67,894 | `video-features/ucf/*.npy` |
| XD-Violence | 800 | 141,650 | `video-features/xd/*.npy` |

Each array keeps the source video stem. `SHA256SUMS.features` lists the expected hash of every array if
you want to verify a copy obtained elsewhere.

## Setup

Use Python 3.11 and install the runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

Scoring the feature arrays does not load the vision-language model. Feature extraction and text encoding
additionally require a local Qwen3-VL-Embedding-2B model directory containing
`scripts/qwen3_vl_embedding.py`. Model weights and raw videos are not part of this repository.

## Reproduce scores and ablations

Place the feature arrays as described above (or set `UCF_FEATURES`/`XD_FEATURES`), then run:

```bash
PYTHON=python3 GPU=0 MODE=ablation DATASET=all bash run.sh
```

The default ablation run scores all 16 combinations of C, V, T, and E, including `off` and the full `CVTE`
configuration. It writes one score array per video and `evaluation.json` for each dataset. The evaluation
JSON includes both paper tables: `leave_one_out` and `factorial_fixed_e`. The fixed-E table contains `E`,
`CE`, `VE`, `TE`, `CVE`, `CTE`, `VTE`, and `CVTE`.

For the full model alone:

```bash
MODE=main DATASET=all GPU=0 bash run.sh
```

Set `DATASET=ucf` or `DATASET=xd` for one dataset. Set `GPU=1` for the second physical GPU. Set
`DEVICE=cpu` for CPU scoring. `PYTHON`, `OUTPUT_ROOT`, `UCF_FEATURES`, `XD_FEATURES`, and `CONFIG`
override the interpreter, output directory, feature directories, and settings file. Each run writes to a
new timestamped directory under `results/` unless `OUTPUT_ROOT` is supplied.

The score phase reads only the feature arrays, prompt descriptions, cached text embeddings, video lists,
and `data/config.json`. The evaluate phase reads `data/{ucf,xd}_annotations.txt` after all video score
files exist. Annotated frame intervals map to windows with `start // 16 : end // 16`. Scores and mapped
labels from all test videos are concatenated before `roc_auc_score` and `average_precision_score`. The
same mapping and aggregation are used by the released
[PRISM implementation](https://github.com/ytC2026/ICML2026-PRISM).

## Expected full-set results

| Configuration | UCF AUC | XD AUC | XD AP |
|---|---:|---:|---:|
| All modules off | 88.8478% | 93.7913% | 79.0752% |
| VENMA (CVTE) | 90.5453% | 94.0588% | 81.3249% |

Exact metrics for every ablation are in `results/expected_metrics.json`. Evaluation also reports AP on
UCF and SNR on both datasets; the manuscript reports UCF AUC and XD AUC/AP.

## Rebuild video features

Video extraction follows the released PRISM protocol: sample every 16th source frame, form
seven-sampled-frame windows at stride one, send
`Represent the video content for anomaly detection.` to the Qwen video encoder, and use
`max_pixels=600000`. UCF uses batch size two; XD uses batch size one. The code reads windows
incrementally while keeping the encoded requests unchanged.

```bash
python feature-extraction/extract_video.py --dataset ucf \
  --model-dir /path/to/Qwen3-VL-Embedding-2B \
  --video-root /path/to/UCF-test-videos \
  --output-root video-features/ucf

python feature-extraction/extract_video.py --dataset xd \
  --model-dir /path/to/Qwen3-VL-Embedding-2B \
  --video-root /path/to/XD-test-videos \
  --output-root video-features/xd
```

The extractor supports `--gpu`, `--rank`, `--world-size`, and `--overwrite`. UCF input can have nested
directories; XD input is a flat video directory. Output `.npy` files retain the source video stem. The
scorer accepts flat or nested feature directories.

## Rebuild text embeddings

The included text banks reproduce the reported scores without re-encoding. To create a new bank from the
included final prompt descriptions, use a separate output path:

```bash
python feature-extraction/encode_text.py --dataset ucf \
  --model-dir /path/to/Qwen3-VL-Embedding-2B \
  --output /path/to/ucf_text_bank_cpu.npz

python feature-extraction/encode_text.py --dataset xd \
  --model-dir /path/to/Qwen3-VL-Embedding-2B \
  --output /path/to/xd_text_bank_cpu.npz
```

The encoder uses the final UCF `effective_prompts` order and the XD class-code order in the supplied JSON.
The prompt prefixes match those used to create the cached embeddings. The scorer checks the description
SHA256 and the number of vectors in each text bank.

## Settings

Both datasets use `data/config.json`. The default projection temperature is 0.3, Gaussian smoothing
standard deviation is 12, and covariance ridge is 0.01. C uses slope 32 and video coefficient 0.20.
V uses rank slope 2.50 and center weight 0.20. T uses seven normal-reference windows and a 5% normal-prompt
trim. E uses weight 0.15. The remaining candidate and reference settings are listed in the config file.
