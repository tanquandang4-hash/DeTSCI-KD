# DeT-SCI

**Video snapshot compressive imaging (SCI) reconstruction with deformable 3D convolutions and temporal attention, plus a knowledge-distillation recipe for lightweight deployment.**

Built on the CACTI engineering stack (registry + config files + dataset pipelines), this repository ships two reconstruction networks:

- **`DeTSCI`** — teacher / single-model baseline, `dim=128`, about **3.60M** parameters;
- **`DeTSCIStudent`** — lightweight student, `dim=96`, about **2.19M** parameters, distilled from the teacher with output, feature and attention terms.

On the six grayscale simulation scenes (`cr=8`, 256×256), the teacher averages **35.30 dB / 0.9687**; the fine-tuned distilled student averages **34.25 dB / 0.9620**.

---

## Table of Contents

- [1. Method Overview](#1-method-overview)
- [2. Repository Layout](#2-repository-layout)
- [3. Installation](#3-installation)
- [4. Data Preparation](#4-data-preparation)
- [5. Quick Start](#5-quick-start)
- [6. Configuration Reference](#6-configuration-reference)
- [7. Results](#7-results)
- [8. Ablation Studies](#8-ablation-studies)
- [9. Known Issues and Caveats](#9-known-issues-and-caveats)
- [10. Adding a Model or Dataset](#10-adding-a-model-or-dataset)
- [11. Acknowledgements](#11-acknowledgements)

---

## 1. Method Overview

### 1.1 Problem Setting

Video SCI compresses `T` high-speed frames into a single 2D measurement `y` through a coded aperture `Φ`:

```
y = Σ_t Φ_t ⊙ x_t
```

The network receives `(y, Φ, Φ_s)` and reconstructs the `T` frames. `Φ_s` is the per-pixel sum of the aperture (used for normalized initialization) and is produced by `generate_masks()` in `cacti/utils/mask.py`.

Inside the model, the measurement is first back-projected into an initial estimate:

```python
meas_re = y / Phi_s            # y: [B,1,H,W], Phi_s: [B,1,H,W]
maskt   = Phi * meas_re        # Phi: [B,T,H,W]; broadcast -> [B,T,H,W]
x       = meas_re + maskt      # [B,T,H,W] -> unsqueeze(1) -> [B,1,T,H,W] fed to the network
```

For the color branch (`color_channels=3`), `bayer_init()` splits the Bayer mosaic into its four 2×2 phases, stacks them along the batch dimension, runs the same initialization per phase, and reassembles the full-resolution grid.

### 1.2 Network Architecture

A single-scale U-shaped topology; every block operates on 5D tensors `[B, C, T, H, W]`:

| Stage | Module | Description |
|-------|--------|-------------|
| Stem | `input1` | `Conv3d(1→dim, 3×3)` + `LeakyReLU`, then `Conv3d(dim→2dim, (1,3,3), stride=(1,2,2))`: **H/W halved, channels doubled** |
| Encoder | `StageBlock` + `Downsample` | `stage_num` stages; `Downsample` is in fact a `stride=1` `(1,3,3)` conv (resolution unchanged); its output is cached in `fea_list` for skip connections |
| Bottleneck | `StageBlock` | `depth_num[-1]` `DenseRNetBlock`s |
| Decoder | `Upsample` + `Fusion` + `StageBlock` | `Upsample` is a `1×1` conv; `Fusion` uses a `1×1` conv to squeeze the concatenated skip (`2dim`) back to `dim` |
| Head | `out1` | `1×1` conv → `Rearrange` → **`PixelShuffle(2)` (2× upsample)** → `Rearrange` → `Conv3d(dim//2 → color_channels, 3×3)` |

> Note: `Downsample` / `Upsample` are only variable names — the former does not downsample and the latter does not upsample. The only spatial rescaling happens in the stem (2× down) and the head (2× up). Apart from the H/W factor, encoder and decoder stages share the same resolution.

Core building blocks:

- **`DeTBlock`** — `ResBlock_d3d` (two `DeformConvPack_d` layers with `dimension='THW'`, residual) + **`TimesAttention3D`** (per-pixel multi-head self-attention across time: `b c d h w → (b h w) d c`, qkv projected to `dim/2`, output projected back to `dim`) + **`FFN`** (channel split + gated conv: `f2(x2 * GELU(f1(x1)))`, fused by a `1×1` conv).
- **`DenseRNetBlock`** — splits input channels by `split_ratio=0.5`; only the first half is partitioned into `group_num` groups and processed with dense cascading (`cf_in = input_list[i] + cf_out`); the result is concatenated with the untouched half, passed through a `1×1` conv, and added back as a residual.
- **`StageBlock`** — stacks `depth` `DenseRNetBlock`s with `group_num = dim // 64` (teacher `dim=256` → 4 groups of 32 channels; student `dim=192` → 3 groups).

`DeTSCIStudent` reuses the same macro topology (stem / encoder / bottleneck / decoder / head) with `dim=96` by default, 2 attention heads, and an optional `use_dcn` switch (when disabled, both convs inside `ResBlock` fall back to plain `Conv3d`). Its `Student*` blocks are defined inline in `__init__`. The `export_attn` flag keeps the last attention map in `last_attn` for distillation.

Both models return a **list** from `forward(y, Phi, Phi_s)` (`out_list`); take `out_list[-1]` as the final reconstruction — the list is there to leave room for multi-stage outputs.

| Tensor | Shape | Meaning |
|--------|-------|---------|
| `y` | `[B, 1, H, W]` | Compressed measurement frame (grayscale) |
| `Phi` | `[B, T, H, W]` | Coded aperture, `T = cr` |
| `Phi_s` | `[B, 1, H, W]` | Per-pixel aperture sum (zeros replaced by 1 to avoid division by zero) |
| Output (grayscale) | `[B, T, H, W]` | Reconstructed `T` frames |
| Output (color) | `[B, 3, T, H, W]` | Reconstructed Bayer RGB volumes |

### 1.3 Knowledge Distillation (Teacher → Student)

`KDLoss` in `cacti/utils/kd_losses.py` combines three terms:

| Term | Computation | Default weight |
|------|-------------|----------------|
| Output distillation `l_out` | Point-wise loss between student and teacher outputs | `lambda_out=1.0` |
| Feature distillation `l_feat` | Alignment on `encoder_last` / `bottleneck` / `decoder_last` (any subset via `feat_pairs`); mismatched channel counts go through `FeatureAdapter` (a `1×1` Conv3d) | `lambda_feat=0.3` |
| Attention distillation `l_attn` | Loss on attention maps after averaging over the head dimension; the student has 2 heads and the teacher 4, so averaging makes them comparable | `lambda_attn=0.1` |

The base loss is selectable via `loss_type`: `mse` / `smoothl1` / `charbonnier`.

Distillation weights follow an optional **multi-phase schedule** (`kd.schedule`):

```
epoch 0 ~ warmup            : output distillation only (feat / attn weights are 0)
epoch warmup ~ warmup+ramp  : feat / attn weights ramp linearly to their target values
epoch ≥ drop_feat_at_epoch  : feature weight drops to late_phase.lambda_feat
```

The total objective is `sqrt(criterion(student_out, gt)) + kd_loss`, i.e. the supervised term is in RMSE form.

Feature hooks (`tools/DDPtrain_amp.py`) are attached to `encoder[-1][0]` (`encoder_last`), `bottleneck`, and `decoder[-1][2]` (`decoder_last`).

---

## 2. Repository Layout

```
DeT_SCI_kd/
├── cacti/
│   ├── datasets/
│   │   ├── davis.py                 # DAVIS training set (slices by mask frame count)
│   │   ├── six_gray_sim_data.py     # Grayscale simulation test set (.mat)
│   │   ├── real_data.py             # Real captured measurements
│   │   ├── davis_bayer.py / matlab_bayer.py / birnat_davis.py
│   │   └── pipelines/               # Augmentation, gray / Bayer measurement synthesis
│   ├── models/
│   │   ├── builder.py               # MODELS registry + build_model
│   │   ├── DeT_SCI.py               # DeTSCI / DeTSCIStudent and building blocks
│   │   └── __init__.py              # Re-exports models (triggers registration)
│   └── utils/                       # config / mask / metrics / kd_losses / logger / optim / loss
├── configs/
│   ├── _base_/                      # Dataset, runtime and default optimizer bases
│   └── DeT-SCI/                     # Main experiment configs + ablation/
├── dcn/                             # Deformable 3D convolution CUDA extension (D3D)
├── docs/ablation_study_plan.md      # Ablation plan and result-recording tables
├── scripts/train_amp.sh             # Multi-GPU AMP training example script
├── tools/                           # Training / testing entry points
├── checkpoints/                     # Pretrained weights
├── test_datasets/                   # Masks, simulation set, real measurements, middle_scale
└── work_dirs/                       # Logs, TensorBoard, weights and reconstructions
```

Entry points under `tools/`:

| Script | Purpose |
|--------|---------|
| `DDPtrain_amp.py` | **Recommended** trainer: AMP + DCN FP32 protection + KD scheduling + gradient clipping + cosine LR |
| `DDPtrain.py` | FP32 training + KD (no AMP, no schedule) |
| `DDPtrain_improved.py` | FP32 training with LR scheduling and seed control |
| `test.py` | Simulation evaluation (PSNR/SSIM + reconstructed frames) |
| `test_amp.py` | Same, with AMP inference (use this for AMP-trained checkpoints) |
| `test_real_data.py` / `test_realamp.py` | Reconstruction from real captures (no GT; images and runtime only) |

---

## 3. Installation

### 3.1 Dependencies

- Python 3.8 / 3.11 (the repo carries both `cpython-38` and `cpython-311` artifacts under `__pycache__` and `dcn/build/`)
- PyTorch + torchvision matching your CUDA version (CUDA ≥ 11.x recommended for AMP training)
- `einops`, `scipy`, `addict`, `opencv-python`, `tensorboard`, `thop` (optional, for FLOPs)

```bash
conda create -n detsci python=3.8 -y
conda activate detsci
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install einops scipy addict opencv-python tensorboard tensorboardX thop
```

### 3.2 Build the deformable convolution extension (required)

`DeT_SCI.py` starts with `from dcn.modules.deform_conv import *`, which needs the compiled `D3D` extension:

```bash
cd dcn
python setup.py build install     # On Windows you can also run dcn\make.bat
cd ..
```

On Linux use `bash dcn/make.sh` (same command inside). Verify the build:

```bash
python -c "from dcn.modules.deform_conv import DeformConvPack_d; print('DCN ok')"
```

> `DeTSCIStudent(use_dcn=False)` does not use the DCN op at runtime, but the model module still imports it, so the extension must remain importable.

### 3.3 About the bundled `venv`

The `venv/` folder points at a Python interpreter that no longer exists (`D:\anaconda3\python.exe`), so invoking it fails with `No Python at ...`. Create your own environment as described in 3.1, or rebuild that virtualenv before use.

---

## 4. Data Preparation

### 4.1 Training data (DAVIS)

`configs/_base_/davis.py` hardcodes a **sample absolute path** for `train_data.data_root`; point it at your local DAVIS frames:

```python
train_data = dict(
    type="DavisData",
    data_root="/path/to/DAVIS/JPEGImages/480p",   # each sub-folder = a consecutive frame sequence
    mask_path="test_datasets/mask/random_mask.mat",
    pipeline=train_pipeline,
    gene_meas=dict(type='GenerationGrayMeas'),
    mask_shape=None,
)
```

`DavisData` walks every video sub-folder under `data_root` and slices consecutive `cr`-frame samples, where `cr` is the number of mask frames. Each frame's **Y channel** (BGR→YCrCb) is used as GT; the measurement is produced by `GenerationGrayMeas` as the mask-weighted sum over frames.

### 4.2 Coded aperture (mask)

| File | Purpose |
|------|---------|
| `test_datasets/mask/random_mask.mat` | Random mask for training and grayscale simulation (`mask` key, shape `[H, W, T]`) |
| `test_datasets/mask/real_mask.mat` | Calibrated mask of the real capture setup (`cr=10`) |
| `test_datasets/mask/mid_color_mask.mat` | Color / middle-scale experiments |

`generate_masks()` crops a random patch of size `train_data.mask_shape=(h, w, cr)` from the full mask; training configs use **128×128, `cr=8`**. The number of mask frames defines the compression ratio `cr`, which is also the temporal length `T` of the network output.

### 4.3 Simulation test set

`test_datasets/simulation/*.mat`. `SixGraySimData` tries the keys `orig` / `patch_save` / `p1` / `p2` / `p3` in order, normalizes by 255, and groups frames by `cr`:

```
aerial32_cacti.mat, crash32_cacti.mat, drop8_cacti.mat,
kobe_cacti.mat, runner8_cacti.mat, traffic_cacti.mat
```

### 4.4 Real captures

`test_datasets/real_data/cr{10,20,30,40,50}/` holds `meas_*.mat` files (key `meas`). `GrayRealData` loads them, divides by 255 and multiplies by `cr/2`. There is no GT for real captures, so the test scripts only write reconstructed frames and report runtime.

Directory convention:

```
test_datasets/
├── mask/            # random_mask.mat / real_mask.mat / mid_color_mask.mat
├── simulation/      # grayscale simulation .mat files
├── middle_scale/    # *_bayer.mat (color / middle-scale experiments)
└── real_data/
    └── cr10..cr50/  # real compressed measurements
```

---

## 5. Quick Start

### 5.1 Training (single GPU)

```bash
python tools/DDPtrain_amp.py configs/DeT-SCI/DeTSCI_base.py \
    --work_dir work_dirs/DeTSCI_base \
    --seed 42
```

### 5.2 Training (multi-GPU DDP)

Distributed mode is detected automatically from the environment variables that `torchrun` injects (`RANK` / `WORLD_SIZE`); you do **not** pass `--distributed`:

```bash
torchrun --nproc_per_node=2 --master_port=3278 \
    tools/DDPtrain_amp.py configs/DeT-SCI/student_finetune.py \
    --work_dir work_dirs/student_finetune \
    --seed 42
```

Resume from a checkpoint:

```bash
python tools/DDPtrain_amp.py configs/DeT-SCI/student_finetune.py \
    --resume work_dirs/student_finetune/checkpoints/epoch_50.pth
```

`--resume` restores the optimizer, scheduler, AMP `GradScaler` state and `best_psnr`. To merely warm-start from weights, set the `checkpoints` field in the config (`Config` supports `_base_` inheritance).

### 5.3 Training outputs

Everything lands in `work_dirs/<config name>/` (override with `--work_dir`):

| Directory | Content |
|-----------|---------|
| `log/` | Text logs (per-epoch avg loss / sup / kd) |
| `show/` | TensorBoard: `loss/total`, `loss/supervised`, `loss/kd_total`, `lr`, `amp/loss_scale` |
| `checkpoints/` | `epoch_*.pth` (epoch / model / optim / scheduler / scaler / best_psnr) and `best_checkpoint.pth` |
| `train_images/` | Periodic side-by-side reconstruction snapshots |
| `test_images/`, `test_log/` | Validation reconstructions and metric logs |

```bash
tensorboard --logdir work_dirs/student_finetune/show
```

### 5.4 Evaluation

```bash
# Simulation data (FP32 inference)
python tools/test.py configs/DeT-SCI/DeTSCI_base.py \
    --weights "checkpoints/epoch_118(teacher).pth"

# Simulation data (AMP inference, for AMP-trained models)
python tools/test_amp.py configs/DeT-SCI/student_finetune.py \
    --weights work_dirs/student_finetune/checkpoints/best_checkpoint.pth

# Real captures (no GT; images only)
python tools/test_realamp.py configs/DeT-SCI/student_real.py \
    --weights checkpoints/epoch_209.pth \
    --work_dir work_dirs/student_real
```

When `--weights` is omitted, the `checkpoints` field of the config is used. The scripts print per-scene PSNR/SSIM, the mean, and average per-frame runtime, and write reconstructions to `work_dirs/<config>/test_images/<scene>/`.

---

## 6. Configuration Reference

All configs are loaded by `Config.fromfile()` and support `_base_` list inheritance (see `cacti/utils/config.py`). The bases live in `configs/_base_/`:

| Base | Role |
|------|------|
| `default_runtime.py` | `optimizer` (Adam 1e-4), `loss` (MSELoss), `runner.max_epochs=200`, logging / snapshot / checkpoint intervals |
| `davis.py` | Training data root, `train_pipeline` (RandomResize / Crop / Flip / Resize), `GenerationGrayMeas` |
| `six_gray_sim_data.py` | Simulation test set `SixGraySimData` |
| `real_data.py` | Real capture dataset and `real_mask.mat` |
| `matlab_bayer.py` | Color / Bayer pipeline |

Main experiment configs under `configs/DeT-SCI/`:

| Config | Model | Key settings |
|--------|-------|--------------|
| `DeTSCI_base.py` | `DeTSCI`, `dim=128` | Teacher baseline, `kd.enabled=False`, `checkpoints=checkpoints/epoch_118.pth` |
| `DeTSCI_KD_student.py` | `DeTSCIStudent`, `dim=108` | Full KD (feat 0.20 / attn 0.08, all three `feat_pairs`) |
| `student_amp_v2.py` | `DeTSCIStudent`, `dim=96` | AMP + tuned KD schedule (warmup 10 / ramp 50 / late 150), `max_epochs=200` |
| `student_finetune.py` | `DeTSCIStudent`, `dim=96` | Low-LR fine-tuning (lr 5e-5, `min_lr=1e-8`, 250 epochs, `grad_clip=1.0`) |
| `DeTSCI_real_cr10.py` | `DeTSCI`, `dim=128` | Real-capture teacher training at `cr=10` (`eval.flag=False`) |
| `student_real.py` | `DeTSCIStudent`, `dim=96` | Real-capture student training at `cr=10` (`real_data.data_root=test_datasets/real_data/cr10`) |

Field cheat sheet:

```python
model = dict(
    type='DeTSCIStudent',   # registry names: DeTSCI / DeTSCIStudent
    color_channels=1,       # 1 = grayscale, 3 = Bayer color
    dim=96,                 # base channels (doubled after the stem)
    stage_num=1,            # encoder/decoder stages; depth_num must match
    depth_num=[2, 1],       # StageBlock depths per stage (last entry goes to the bottleneck)
    num_heads=2,            # student only (teacher DeTBlock is fixed at 4 heads)
    export_attn=True,       # cache attention maps for distillation
    use_dcn=True,           # student default is False; configs opt in explicitly
)

amp = dict(enabled=True, opt_level='O1', loss_scale='dynamic')   # selects the AMP branch
kd  = dict(enabled=True, teacher=..., kd_loss=..., feat_pairs=[...],
           attn_collect=True, schedule=dict(...))
```

---

## 7. Results

Grayscale simulation set (`test_datasets/simulation`, `cr=8`, 256×256). Numbers come from the most recent test logs under `work_dirs/*/test_log`, corresponding to the checkpoints referenced by each config:

| Model | aerial32 | crash32 | drop8 | kobe | runner8 | traffic | **Mean PSNR** | **Mean SSIM** |
|-------|---------|---------|-------|------|---------|---------|---------------|---------------|
| `DeTSCI` (teacher, 3.60M) | 31.14 | 30.40 | 43.93 | 34.57 | 41.04 | 30.71 | **35.30** | **0.9687** |
| `DeTSCIStudent` + KD (`student_amp_v2`) | 30.25 | 29.54 | 42.43 | 32.94 | 38.90 | 28.98 | **33.84** | **0.9584** |
| `DeTSCIStudent` + KD + fine-tune (`student_finetune`) | 30.55 | 29.82 | 42.70 | 33.36 | 39.57 | 29.48 | **34.25** | **0.9620** |

The student carries about **61%** of the teacher's parameters (2.19M vs 3.60M) and, after fine-tuning, trails it by roughly **1.05 dB** mean PSNR.

---

## 8. Ablation Studies

The full plan, result-recording tables and paper-writing notes are in [`docs/ablation_study_plan.md`](docs/ablation_study_plan.md). Configs live in `configs/DeT-SCI/ablation/`:

| Group | Configs | Variable |
|-------|---------|----------|
| A: distillation components | `A0_no_kd.py` / `A1_out_only.py` / `A2_out_feat.py` / `A3_out_attn.py` | `kd.enabled`, `lambda_feat`, `lambda_attn` |
| B: feature alignment position | `B1_feat_bottleneck.py` / `B2_feat_decoder.py` / `B4_feat_all.py` | `feat_pairs` |
| C: student architecture | `C1_no_dcn.py` | `use_dcn=False` |
| D: distillation schedule | `D1_no_schedule.py` | `schedule.enabled=False` |
| E: base loss type | `E1_mse.py` / `E3_charbonnier.py` | `kd_loss.loss_type` |

Template (A0, the no-distillation baseline):

```bash
python tools/DDPtrain_amp.py configs/DeT-SCI/ablation/A0_no_kd.py \
    --work_dir work_dirs/ablation_A0 --seed 42
```

> `docs/ablation_study_plan.md` mentions `C2_dim64`, `C3_shallow`, `C4_heads4`, `D2_no_warmup`, `D3_no_late_drop` and `E2`, but those configs are **not present** in the repo; derive them from the existing files in the same folder if needed.

---

## 9. Known Issues and Caveats

1. **Teacher checkpoint name mismatch** — `checkpoints/` actually contains `epoch_118(teacher).pth`, while several configs reference `checkpoints/epoch_118.pth`. Rename the file or update `kd.teacher.checkpoints` before running.
2. **Some configs point at missing weights** — `DeTSCI_KD_student.py` expects `checkpoints/epoch_97.pth` and `student_amp_v2.py` expects `checkpoints/epoch_179.pth`, but only `epoch_118(teacher).pth`, `epoch_199.pth` and `epoch_209.pth` exist. Set `checkpoints=None` to train from scratch.
3. **KD scheduling only works in the AMP branch** — in the FP32 branch of `tools/DDPtrain_amp.py` the scale computation is a placeholder (`pass`), so `scales` stays `None` and `kd.schedule` has no effect. Keep `amp.enabled=True` if you want progressive distillation.
4. **AMP + DCN requires the FP32 hooks** — `register_amp_hooks_for_dcn()` wraps every `DeformConv*` `forward` so inputs are cast to FP32, `autocast` is disabled, and outputs are cast back to FP16. Keep this intact if you refactor the trainer; otherwise DCN will raise `not implemented for 'Half'` in fp16.
5. **`scripts/train_amp.sh` is not runnable as-is** — it points to a non-existent `configs/DeT-SCI/speed/student_amp.py` and passes `--distributed`, which `parse_args()` never defines (argparse will fail with `unrecognized arguments`). Use the `torchrun` command from Section 5 instead.
6. **Training data path is hardcoded** — `configs/_base_/davis.py` ships a sample absolute `data_root`; update it for your machine or dataset construction fails immediately.
7. **`DeTSCIStudent` defaults to `use_dcn=False`** — every shipped config sets `use_dcn=True` explicitly; don't forget it in new configs.
8. **The supervised term is RMSE** — the trainer computes `torch.sqrt(criterion(out, gt))`, which is RMSE only when `loss=MSELoss`. With `SmoothL1` this is no longer a standard RMSE; keep that in mind when tuning.
9. **The evaluation flag is `--weights`** — `docs/ablation_study_plan.md` still shows `--checkpoint <path>`, which `tools/test.py` does not accept.
10. **No `LICENSE` file** — consider adding one before publishing, and comply with the licenses of the upstream CACTI project and `D3D` (D3Dnet).

---

## 10. Adding a Model or Dataset

Components are wired through registries (`cacti/utils/registry.py`); registering and importing is all it takes.

New model:

```python
# cacti/models/MyModel.py
from .builder import MODELS
import torch.nn as nn

@MODELS.register_module
class MyModel(nn.Module):
    def __init__(self, color_channels, dim=64, **kwargs):
        super().__init__()
        ...

    def forward(self, y, Phi, Phi_s):
        ...
        return [out]   # convention: return a list
```

Then add the import to `cacti/models/__init__.py`:

```python
from .MyModel import MyModel
```

After that, `model=dict(type='MyModel', ...)` in a config is enough for `build_model()` to construct it. Note that `register_module` is used bare (no parentheses) and keys the registry by `cls.__name__`.

Datasets follow the same pattern: `@DATASETS.register_module` plus an import in `cacti/datasets/__init__.py`. `__getitem__` must return `(gt, meas)` for training and `(meas, gt)` for evaluation (matching `meas, gt = data` in `tools/test.py`). Augmentation operators register in `PIPELINES` inside `cacti/datasets/pipelines/builder.py`.

New losses: `@LOSSES.register_module` in `cacti/utils/loss_builder.py`; `KDLoss` is already registered in `cacti/utils/kd_losses.py` and auto-loaded when `loss_builder` is imported.

---

## 11. Acknowledgements

- The engineering stack, data pipelines and SCI imaging formulation follow the open-source **CACTI** implementation.
- The deformable 3D convolution operator (`dcn/`, package name `D3D`) comes from [D3Dnet](https://github.com/XinyiYing/D3Dnet).

If you use this repository in research, please cite the above works as well.
