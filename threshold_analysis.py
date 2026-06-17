"""
threshold_analysis.py

Evaluates the classifier at different confidence thresholds and produces:
  - face_threshold_analysis.csv      : overall face head metrics per threshold
  - face_per_class_threshold.csv     : TP and FN broken down per folder per threshold
  - mask_threshold_analysis.csv      : overall mask head metrics per threshold
  - mask_per_class_threshold.csv     : TP and FN broken down per folder per threshold

Usage:
    python threshold_analysis.py \
        --onnx_path  "C:/Users/smaruboina/.insightface/models/buffalo_s/w600k_mbf.onnx" \
        --heads_onnx classifier_heads.onnx \
        --img_dir    data/val \
        --output_dir reports/
"""

import argparse
import csv
import os

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#  Preprocessing
# --------------------------------------------------------------------------- #

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    img = img_bgr.astype(np.float32)
    img = (img - 127.5) / 127.5
    img = img[:, :, ::-1]
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0)
    return np.ascontiguousarray(img)


# --------------------------------------------------------------------------- #
#  Build ORT sessions
# --------------------------------------------------------------------------- #

def build_sessions(backbone_onnx: str, heads_onnx: str):
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    backbone = ort.InferenceSession(backbone_onnx,
                                    sess_options=sess_options,
                                    providers=providers)
    heads    = ort.InferenceSession(heads_onnx,
                                    sess_options=sess_options,
                                    providers=providers)
    print(f"[sessions] Provider : {backbone.get_providers()[0]}")
    return backbone, heads


# --------------------------------------------------------------------------- #
#  Collect predictions — stores per-folder results separately
# --------------------------------------------------------------------------- #

def collect_predictions(backbone, heads, val_dir: str):
    """
    Returns:
        per_folder : dict of folder_name ->
                        {
                          "face_probs"  : np.array,
                          "face_label"  : int (1 or 0),
                          "mask_probs"  : np.array (empty if mask_label == -1),
                          "mask_label"  : int (1, 0, or -1)
                        }
    """
    bb_in  = backbone.get_inputs()[0].name
    bb_out = backbone.get_outputs()[0].name
    h_in   = heads.get_inputs()[0].name

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    DIRS = {
        "face":     (1, -1),
        "non-face": (0, -1),
        "mask":     (1,  1),
        "non-mask": (1,  0),
    }

    per_folder = {}

    for folder, (fl, ml) in DIRS.items():
        folder_path = os.path.join(val_dir, folder)
        if not os.path.exists(folder_path):
            print(f"  [skip] {folder_path} not found")
            continue

        imgs = [f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in IMG_EXTS]

        print(f"  {folder:12s}: {len(imgs):>8,} images  (face_label={fl}, mask_label={ml})")

        face_probs_list = []
        mask_probs_list = []

        for fname in tqdm(imgs, desc=f"  {folder}", leave=False):
            img_bgr = cv2.imread(os.path.join(folder_path, fname))
            if img_bgr is None:
                continue
            if img_bgr.shape[:2] != (112, 112):
                img_bgr = cv2.resize(img_bgr, (112, 112))

            x   = preprocess(img_bgr)
            emb = backbone.run([bb_out], {bb_in: x})[0]

            face_logits, mask_logits = heads.run(
                ["face_logits", "mask_logits"], {h_in: emb}
            )

            def softmax(v):
                e = np.exp(v - v.max())
                return e / e.sum()

            face_probs_list.append(float(softmax(face_logits[0])[1]))
            if ml != -1:
                mask_probs_list.append(float(softmax(mask_logits[0])[1]))

        per_folder[folder] = {
            "face_probs" : np.array(face_probs_list),
            "face_label" : fl,
            "mask_probs" : np.array(mask_probs_list),
            "mask_label" : ml,
        }

    return per_folder


# --------------------------------------------------------------------------- #
#  Metrics at a given threshold
# --------------------------------------------------------------------------- #

def metrics_at_threshold(probs, labels, threshold):
    preds = (probs >= threshold).astype(int)
    tp = int(np.sum((preds == 1) & (labels == 1)))
    tn = int(np.sum((preds == 0) & (labels == 0)))
    fp = int(np.sum((preds == 1) & (labels == 0)))
    fn = int(np.sum((preds == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    fpr       = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr       = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    accuracy  = (tp + tn) / len(labels) if len(labels) > 0 else 0.0

    return {
        "threshold" : round(threshold, 2),
        "accuracy"  : round(accuracy,  4),
        "precision" : round(precision, 4),
        "recall"    : round(recall,    4),
        "f1"        : round(f1,        4),
        "fpr"       : round(fpr,       4),
        "fnr"       : round(fnr,       4),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


# --------------------------------------------------------------------------- #
#  Per-class TP / FN at each threshold
# --------------------------------------------------------------------------- #

def per_class_breakdown(per_folder, prob_key, label_key, thresholds, positive_label=1):
    """
    For each folder that has the given label as the positive class,
    compute TP and FN at every threshold.

    Returns list of rows:
        { "threshold": t, "folder_A_tp": x, "folder_A_fn": y, ... }
    """
    # Only include folders where label matches positive_label
    relevant = {
        name: data
        for name, data in per_folder.items()
        if data[label_key] == positive_label and len(data[prob_key]) > 0
    }

    if not relevant:
        return []

    rows = []
    for t in thresholds:
        row = {"threshold": round(t, 2)}
        for folder_name, data in relevant.items():
            probs  = data[prob_key]
            preds  = (probs >= t).astype(int)
            tp     = int(np.sum(preds == 1))   # predicted positive = TP (since all are positive class)
            fn     = int(np.sum(preds == 0))   # predicted negative = FN
            total  = len(probs)
            recall = round(tp / total, 4) if total > 0 else 0.0
            row[f"{folder_name}_total"]  = total
            row[f"{folder_name}_tp"]     = tp
            row[f"{folder_name}_fn"]     = fn
            row[f"{folder_name}_recall"] = recall
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
#  Save CSV
# --------------------------------------------------------------------------- #

def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved → {path}")


# --------------------------------------------------------------------------- #
#  Print table
# --------------------------------------------------------------------------- #

def print_table(rows, head_name):
    print(f"\n{'='*75}")
    print(f"  {head_name}")
    print(f"{'='*75}")
    print(f"  {'Thresh':>6}  {'Acc':>6}  {'Prec':>6}  {'Recall':>6}  "
          f"{'F1':>6}  {'FPR':>6}  {'FNR':>6}  {'TP':>7}  {'FP':>7}  {'FN':>7}")
    print(f"  {'-'*70}")
    for r in rows:
        print(f"  {r['threshold']:>6.2f}  {r['accuracy']:>6.4f}  "
              f"{r['precision']:>6.4f}  {r['recall']:>6.4f}  "
              f"{r['f1']:>6.4f}  {r['fpr']:>6.4f}  {r['fnr']:>6.4f}  "
              f"{r['tp']:>7}  {r['fp']:>7}  {r['fn']:>7}")


def print_per_class_table(rows, head_name):
    if not rows:
        return
    print(f"\n{'='*75}")
    print(f"  {head_name} — per class TP / FN breakdown")
    print(f"{'='*75}")

    # Get folder names from keys
    folders = []
    for k in rows[0].keys():
        if k.endswith("_tp"):
            folders.append(k.replace("_tp", ""))

    # Header
    header = f"  {'Thresh':>6}"
    for f in folders:
        total = rows[0].get(f"{f}_total", "?")
        header += f"  {f}(n={total})"
        header += f"{'TP':>8}{'FN':>8}{'Recall':>8}"
    print(header)
    print(f"  {'-'*70}")

    for r in rows:
        line = f"  {r['threshold']:>6.2f}"
        for f in folders:
            tp     = r.get(f"{f}_tp", 0)
            fn     = r.get(f"{f}_fn", 0)
            recall = r.get(f"{f}_recall", 0.0)
            line  += f"  {tp:>8}{fn:>8}{recall:>8.4f}"
        print(line)


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx_path",  default="C:/Users/smaruboina/.insightface/models/buffalo_s/w600k_mbf.onnx")
    p.add_argument("--heads_onnx", required=True)
    p.add_argument("--img_dir",    required=True)
    p.add_argument("--output_dir", default="reports")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    backbone, heads = build_sessions(args.onnx_path, args.heads_onnx)

    print(f"\nCollecting predictions from {args.img_dir} ...")
    per_folder = collect_predictions(backbone, heads, args.img_dir)

    thresholds = [0.10, 0.20, 0.30, 0.40, 0.50,
                  0.60, 0.70, 0.80, 0.90, 0.95, 0.99]

    # ── Face / Non-Face overall ──
    all_face_probs  = np.concatenate([d["face_probs"]  for d in per_folder.values()])
    all_face_labels = np.concatenate([
        np.full(len(d["face_probs"]), d["face_label"])
        for d in per_folder.values()
    ])

    face_rows = [metrics_at_threshold(all_face_probs, all_face_labels, t)
                 for t in thresholds]
    print_table(face_rows, "Face / Non-Face Head — overall")
    save_csv(face_rows,
             os.path.join(args.output_dir, "face_threshold_analysis.csv"))

    # ── Face per-class TP/FN (positive folders only) ──
    face_pc_rows = per_class_breakdown(
        per_folder, "face_probs", "face_label", thresholds, positive_label=1
    )
    print_per_class_table(face_pc_rows, "Face / Non-Face Head")
    save_csv(face_pc_rows,
             os.path.join(args.output_dir, "face_per_class_threshold.csv"))

    # ── Mask / Non-Mask overall ──
    mask_folders = {k: v for k, v in per_folder.items() if v["mask_label"] != -1}
    if mask_folders:
        all_mask_probs  = np.concatenate([d["mask_probs"]  for d in mask_folders.values()])
        all_mask_labels = np.concatenate([
            np.full(len(d["mask_probs"]), d["mask_label"])
            for d in mask_folders.values()
        ])
        mask_rows = [metrics_at_threshold(all_mask_probs, all_mask_labels, t)
                     for t in thresholds]
        print_table(mask_rows, "Mask / Non-Mask Head — overall")
        save_csv(mask_rows,
                 os.path.join(args.output_dir, "mask_threshold_analysis.csv"))

        # ── Mask per-class TP/FN ──
        mask_pc_rows = per_class_breakdown(
            mask_folders, "mask_probs", "mask_label", thresholds, positive_label=1
        )
        print_per_class_table(mask_pc_rows, "Mask / Non-Mask Head")
        save_csv(mask_pc_rows,
                 os.path.join(args.output_dir, "mask_per_class_threshold.csv"))

    print(f"\nDone. Reports saved to {args.output_dir}/")
    print(f"  face_threshold_analysis.csv    — overall face head metrics")
    print(f"  face_per_class_threshold.csv   — TP/FN per folder (face, mask, non-mask)")
    print(f"  mask_threshold_analysis.csv    — overall mask head metrics")
    print(f"  mask_per_class_threshold.csv   — TP/FN per folder (mask only)")


if __name__ == "__main__":
    main()