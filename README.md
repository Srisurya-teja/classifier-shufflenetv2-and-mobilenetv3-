# Face & Mask Classifier

A dual-task deep learning system that classifies 112×112 face crops as:
1. **Face vs Non-Face** — is the crop a human face?
2. **Mask vs No-Mask** — if a face, is the person wearing a mask?

Input crops are 112×112 BGR images produced by [RetinaFace](https://github.com/deepinsight/insightface) `norm_crop`, matching the preprocessing pipeline of InsightFace's ArcFace backbone.

---

## Architecture

```
Input (112×112 BGR crop)
        │
        ▼
ShuffleNetV2 x0.5 Backbone
  conv1 → maxpool → stage2 → stage3 → stage4 → conv5 → GAP
        │ (1024-dim → projection → 128-dim embedding)
        ▼
  DualHeadClassifier
  ├── face_head: Linear → PReLU → Dropout(0.3) → 2 logits
  └── mask_head: Linear → PReLU → Dropout(0.3) → 2 logits
```

**Key design choices:**
- ImageNet-pretrained ShuffleNetV2 x0.5 (~1.4M params) adapted for 112×112 input
- Embedding projected from 1024 → 128 dims (Linear + BatchNorm, no bias)
- `mask_label = -1` for non-face images; `CrossEntropyLoss(ignore_index=-1)` skips those samples for the mask head
- Face head uses class weights `[3.0, 1.0]` to compensate for ~3:1 face/non-face imbalance

---

## Data Layout

```
data/
├── train/
│   ├── face/       # face crops — face_label=1, mask_label=-1 (ignored)
│   ├── non-face/   # non-face crops — face_label=0, mask_label=-1 (ignored)
│   ├── mask/       # masked face crops — face_label=1, mask_label=1
│   └── non-mask/   # unmasked face crops — face_label=1, mask_label=0
└── val/            # same four folders
```

~671k total images; all 112×112 BGR PNG/JPEG crops.

---

## Setup

### Local

```bash
pip install -r requirements.txt
```

> `requirements.txt` is UTF-16 LE encoded (Windows-generated). On Linux, use the Dockerfile instead.

### Docker (recommended for Linux / GPU)

```bash
docker build -t face-mask-classifier .
docker run --gpus all -v /path/to/project:/app face-mask-classifier \
    python shufflenet_train.py --data_root data/ ...
```

The Docker image uses CUDA 12.2 + cuDNN 8 and pre-caches ShuffleNetV2 weights so training works offline.

---

## Training

Training uses a two-phase approach to avoid interference between the face and mask tasks.

### Phase 1 — Face/Non-Face only

```bash
python shufflenet_train.py \
    --data_root data/ \
    --save_dir  checkpoints/phase1 \
    --mask_weight 0.0 \
    --epochs 50 --batch_size 256 --lr 1e-3
```

The mask head is frozen during Phase 1. Checkpointed on best validation recall.

### Phase 2 — Mask/Non-Mask only (resume from Phase 1)

```bash
python shufflenet_train.py \
    --data_root data/ \
    --save_dir  checkpoints/phase2 \
    --face_weight 0.0 \
    --resume checkpoints/phase1/best_face_head.pt \
    --epochs 50 --lr 1e-3
```

The face head is frozen during Phase 2. Checkpointed on best validation accuracy.

### All training flags

| Flag | Default | Description |
|---|---|---|
| `--data_root` | required | Path to `data/` directory |
| `--save_dir` | `checkpoints` | Output directory |
| `--embedding_dim` | `128` | Backbone output dimension |
| `--epochs` | `50` | Maximum epochs |
| `--patience` | `7` | Early stopping patience (0 = disabled) |
| `--batch_size` | `256` | Training batch size |
| `--lr` | `1e-3` | Learning rate (AdamW + CosineAnnealingLR) |
| `--face_weight` | `1.0` | Loss weight for face head |
| `--mask_weight` | `1.0` | Loss weight for mask head |
| `--num_workers` | `4` | DataLoader workers |
| `--no_pretrained` | off | Disable ImageNet pretrained weights |
| `--resume` | `None` | Resume from a `.pt` checkpoint |

### Training outputs

Saved to `--save_dir`:

```
checkpoints/
├── best_face_head.pt       # best val checkpoint (face recall)
├── best_mask_head.pt       # best val checkpoint (mask accuracy)
├── backbone.onnx           # exported backbone
├── classifier_heads.onnx   # exported heads
├── classifier_full.onnx    # fused end-to-end model
├── metrics.csv             # epoch-level metrics (train + val)
├── metrics.jsonl           # same metrics in JSONL format
└── train.log               # full training log
```

---

## Export to ONNX

After training, ONNX files are automatically exported. To export manually from existing checkpoints:

```bash
python shufflenet_export_infer.py --mode export \
    --onnx_path   checkpoints/phase2/backbone.onnx \
    --face_ckpt   checkpoints/phase1/best_face_head.pt \
    --mask_ckpt   checkpoints/phase2/best_mask_head.pt \
    --embedding_dim 128 \
    --output      classifier_heads.onnx
```

To optimize an existing fused ONNX graph (saves a pre-optimized ORT graph):

```bash
python shufflenet_export_infer.py --mode optimize \
    --fused_onnx classifier_full.onnx \
    --output     classifier_full_opt.onnx
```

---

## Inference & Benchmarking

### Chained (two ONNX files)

```bash
python shufflenet_export_infer.py --mode benchmark \
    --onnx_path  backbone.onnx \
    --heads_onnx classifier_heads.onnx \
    --img_dir    data/val/face \
    --runs       200
```

### Fused (single ONNX file)

```bash
python shufflenet_export_infer.py --mode benchmark \
    --fused_onnx classifier_full.onnx \
    --img_dir    data/val/face \
    --runs       200
```

**Latency target: P95 ≤ 5 ms** (CPU, single image)

Preprocessing uses `cv2.dnn.blobFromImage` for a single C++ call instead of multiple numpy operations: subtract mean (127.5), scale (1/127.5), BGR→RGB, HWC→NCHW.

---

## Threshold Analysis

Sweeps confidence thresholds from 0.0 to 1.0 in 0.01 increments and writes four CSV reports:

```bash
python threshold_analysis.py \
    --onnx_path  backbone.onnx \
    --heads_onnx classifier_heads.onnx \
    --img_dir    data/val \
    --output_dir reports/
```

**Reports generated:**

| File | Contents |
|---|---|
| `face_threshold_analysis.csv` | Precision, recall, F1, FPR, FNR for face head at each threshold |
| `face_per_class_threshold.csv` | TP/FN breakdown per folder per threshold |
| `mask_threshold_analysis.csv` | Same metrics for mask head |
| `mask_per_class_threshold.csv` | TP/FN breakdown per folder per threshold |

---

## Metrics Tracked

Each epoch logs the following for both face and mask heads:

- Accuracy, Precision, Recall, F1
- False Positive Rate (FPR), False Negative Rate (FNR)
- ROC-AUC
- Confusion matrix (TN/FP/FN/TP)

---

## File Overview

| File | Description |
|---|---|
| `shufflenet_model.py` | `ShuffleNetV2Backbone`, `DualHeadClassifier`, `EndToEndClassifier` |
| `shufflenet_train.py` | Training loop, loss, metrics, CSV/JSONL logging, early stopping |
| `dataset.py` | `FaceMaskDataset` with augmentation, `build_dataloaders()` |
| `shufflenet_export_infer.py` | ONNX export, ORT optimization, chained/fused inference, benchmark |
| `threshold_analysis.py` | Threshold sweep producing four CSV reports |
| `Dockerfile` | CUDA 12.2 + cuDNN 8 image with pre-cached ShuffleNetV2 weights |
