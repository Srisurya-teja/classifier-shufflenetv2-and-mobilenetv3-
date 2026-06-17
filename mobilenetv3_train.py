"""
train_mobilenetv3.py

End-to-end fine-tuning of MobileNetV3 Small backbone with dual classification heads.


  Phase 1 — Face/Non-Face only:
     python train_mobilenetv3.py \
         --data_root   data/ \
         --save_dir    checkpoints/mv3_phase1 \
         --mask_weight 0.0

  Phase 2 — Mask/Non-Mask only:
     python train_mobilenetv3.py \
         --data_root   data/ \
         --save_dir    checkpoints/mv3_phase2 \
         --face_weight 0.0 \
         --resume      checkpoints/mv3_phase1/best_face_head.pt

  After training, exports ONNX files to save_dir/:
     backbone.onnx          <- backbone for production
     classifier_heads.onnx  <- heads ONNX
     best_face_head.pt      <- PyTorch checkpoint

  Merge into single ONNX (existing script):
     python merge_to_single_onnx.py \
         --backbone  checkpoints/mv3_phase2/backbone.onnx \
         --heads     checkpoints/mv3_phase2/classifier_heads.onnx \
         --output    classifier_full_mv3.onnx --verify --benchmark

  Benchmark (existing script):
     python export_and_infer.py --mode benchmark \
         --fused_onnx classifier_full_mv3.onnx \
         --img_dir data/val/face --runs 200
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mobilenetv3_model import (MobileNetV3SmallBackbone,
                               DualHeadClassifier, EndToEndClassifier)
from dataset import build_dataloaders


# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #

def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(save_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
#  CSV logger
# --------------------------------------------------------------------------- #

class CSVLogger:
    FIELDS = [
        "epoch", "split", "loss",
        "face_acc", "face_precision", "face_recall", "face_f1",
        "face_fpr", "face_fnr", "face_auc",
        "mask_acc", "mask_precision", "mask_recall", "mask_f1",
        "mask_fpr", "mask_fnr", "mask_auc",
    ]

    def __init__(self, save_dir: str):
        self.path = os.path.join(save_dir, "metrics.csv")
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def _extract(self, metrics, prefix):
        if metrics is None:
            return {f"{prefix}_{k}": None
                    for k in ["acc","precision","recall","f1","fpr","fnr","auc"]}
        return {
            f"{prefix}_acc"       : round(metrics["accuracy"],  4),
            f"{prefix}_precision" : round(metrics["precision"], 4),
            f"{prefix}_recall"    : round(metrics["recall"],    4),
            f"{prefix}_f1"        : round(metrics["f1"],        4),
            f"{prefix}_fpr"       : round(metrics["fpr"],       4),
            f"{prefix}_fnr"       : round(metrics["fnr"],       4),
            f"{prefix}_auc"       : round(metrics["roc_auc"], 4) if metrics["roc_auc"] else None,
        }

    def write(self, epoch, split, loss, face_metrics, mask_metrics):
        row = {"epoch": epoch, "split": split, "loss": round(loss, 6)}
        row.update(self._extract(face_metrics, "face"))
        row.update(self._extract(mask_metrics, "mask"))
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


# --------------------------------------------------------------------------- #
#  JSONL logger
# --------------------------------------------------------------------------- #

class JSONLLogger:
    def __init__(self, save_dir: str):
        self.path = os.path.join(save_dir, "metrics.jsonl")
        open(self.path, "w").close()

    def write(self, epoch, split, loss, face_metrics, mask_metrics):
        record = {
            "epoch": epoch, "split": split, "loss": round(loss, 6),
            "face_metrics": face_metrics, "mask_metrics": mask_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
#  Early stopping
# --------------------------------------------------------------------------- #

class EarlyStopping:
    def __init__(self, patience: int, logger: logging.Logger):
        self.patience = patience
        self.logger   = logger
        self.counter  = 0
        self.best     = None

    def step(self, metric: float) -> bool:
        if self.patience <= 0:
            return False
        if self.best is None or metric > self.best:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.logger.info(f"  [EarlyStopping] Stopping early")
                return True
        return False


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #



class DualHeadLoss(nn.Module):
    def __init__(self, face_weight=1.0, mask_weight=1.0,
                 face_class_weights=None):
        super().__init__()
        self.face_w = face_weight
        self.mask_w = mask_weight

        # Face head: weight non-face (class 0) higher to fix FPR
        # [3.0, 1.0] compensates for ~3:1 face/non-face imbalance
        if face_class_weights is not None:
            self.face_ce = nn.CrossEntropyLoss(
                weight=torch.tensor(face_class_weights),
                ignore_index=-1
            )
        else:
            self.face_ce = nn.CrossEntropyLoss(ignore_index=-1)

        # Mask head: balanced dataset, no class weights needed
        self.mask_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, face_logits, mask_logits, face_labels, mask_labels):
        device = face_logits.device
        self.face_ce.weight = self.face_ce.weight.to(device) if self.face_ce.weight is not None else None

        f = torch.tensor(0.0, device=device)
        m = torch.tensor(0.0, device=device)
        if self.face_w > 0 and (face_labels != -1).any():
            f = self.face_ce(face_logits, face_labels)
        if self.mask_w > 0 and (mask_labels != -1).any():
            m = self.mask_ce(mask_logits, mask_labels)
        loss = self.face_w * f + self.mask_w * m
        return loss, float(f.detach()), float(m.detach())

# class DualHeadLoss(nn.Module):
#     def __init__(self, face_weight=1.0, mask_weight=1.0):
#         super().__init__()
#         self.face_w = face_weight
#         self.mask_w = mask_weight
#         self.ce = nn.CrossEntropyLoss(ignore_index=-1)

#     def forward(self, face_logits, mask_logits, face_labels, mask_labels):
#         device = face_logits.device
#         f = torch.tensor(0.0, device=device)
#         m = torch.tensor(0.0, device=device)
#         if self.face_w > 0 and (face_labels != -1).any():
#             f = self.ce(face_logits, face_labels)
#         if self.mask_w > 0 and (mask_labels != -1).any():
#             m = self.ce(mask_logits, mask_labels)
#         loss = self.face_w * f + self.mask_w * m
#         return loss, float(f.detach()), float(m.detach())


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(all_preds, all_labels, all_probs, ignore=-1):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    valid  = labels != ignore
    preds, labels, probs = preds[valid], labels[valid], probs[valid]

    if len(labels) == 0:
        return None

    has_both = len(np.unique(labels)) == 2
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy"  : accuracy_score(labels, preds),
        "precision" : precision_score(labels, preds, zero_division=0),
        "recall"    : recall_score(labels, preds, zero_division=0),
        "f1"        : f1_score(labels, preds, zero_division=0),
        "fpr"       : fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr"       : fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "roc_auc"   : roc_auc_score(labels, probs) if has_both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def log_metrics(logger, metrics, head_name):
    if metrics is None:
        logger.info(f"  [{head_name}] Not enough data.")
        return
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    auc = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A"
    logger.info(f"  [{head_name}]")
    logger.info(f"    Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"    Precision : {metrics['precision']:.4f}")
    logger.info(f"    Recall    : {metrics['recall']:.4f}")
    logger.info(f"    F1        : {metrics['f1']:.4f}")
    logger.info(f"    FPR       : {metrics['fpr']:.4f}")
    logger.info(f"    FNR       : {metrics['fnr']:.4f}")
    logger.info(f"    ROC-AUC   : {auc}")
    logger.info(f"    CM: TN={tn} FP={fp} FN={fn} TP={tp}")


# --------------------------------------------------------------------------- #
#  One epoch
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = n = 0
    face_preds, face_labels, face_probs = [], [], []
    mask_preds, mask_labels, mask_probs = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"  {'train' if train else 'val'}",
                    leave=False, unit="batch", dynamic_ncols=True)
        for imgs, face_lbl, mask_lbl in pbar:
            imgs     = imgs.to(device)
            face_lbl = face_lbl.to(device)
            mask_lbl = mask_lbl.to(device)

            face_logits, mask_logits = model(imgs)
            loss, _, _ = criterion(face_logits, mask_logits, face_lbl, mask_lbl)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item(); n += 1
            pbar.set_postfix(loss=f"{total_loss/n:.4f}")

            face_preds  += face_logits.argmax(1).cpu().tolist()
            face_labels += face_lbl.cpu().tolist()
            face_probs  += torch.softmax(face_logits, 1)[:, 1].cpu().tolist()
            mask_preds  += mask_logits.argmax(1).cpu().tolist()
            mask_labels += mask_lbl.cpu().tolist()
            mask_probs  += torch.softmax(mask_logits, 1)[:, 1].cpu().tolist()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (total_loss / n,
            compute_metrics(face_preds, face_labels, face_probs),
            compute_metrics(mask_preds, mask_labels, mask_probs))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True)
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--patience",      type=int,   default=7)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--face_weight",   type=float, default=1.0)
    p.add_argument("--mask_weight",   type=float, default=1.0)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--no_pretrained", action="store_true",
                   help="Disable ImageNet pretrained weights")
    p.add_argument("--resume",        default=None)
    return p.parse_args()


def main():
    args      = parse_args()
    logger    = setup_logger(args.save_dir)
    csv_log   = CSVLogger(args.save_dir)
    jsonl_log = JSONLLogger(args.save_dir)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    # Phase detection
    if args.face_weight > 0 and args.mask_weight == 0.0:
        phase = "Phase 1 — Face/Non-Face only"
    elif args.face_weight == 0.0 and args.mask_weight > 0:
        phase = "Phase 2 — Mask/Non-Mask only"
    else:
        phase = "Full — both heads"

    # ── Build model ──
    backbone = MobileNetV3SmallBackbone(
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
    )
    heads = DualHeadClassifier(embedding_dim=args.embedding_dim)
    model = EndToEndClassifier(backbone, heads).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("=" * 60)
    logger.info(f"  Run started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Backbone        : MobileNetV3 Small")
    logger.info(f"  Phase           : {phase}")
    logger.info(f"  Device          : {device}")
    logger.info(f"  Total params    : {total_params:,}")
    logger.info(f"  Embedding dim   : {args.embedding_dim}")
    logger.info(f"  face_weight     : {args.face_weight}")
    logger.info(f"  mask_weight     : {args.mask_weight}")
    logger.info(f"  Epochs          : {args.epochs}")
    logger.info(f"  Patience        : {args.patience if args.patience > 0 else 'disabled'}")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  LR              : {args.lr}")
    logger.info("=" * 60)

    train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        pin_memory=pin_memory
    )

    # criterion = DualHeadLoss(args.face_weight, args.mask_weight)
    criterion = DualHeadLoss(
            args.face_weight, args.mask_weight,
            face_class_weights=[3.0, 1.0]
        )

    # ── Optimizer — freeze inactive head for single-phase training ──
    if args.face_weight > 0 and args.mask_weight == 0.0:
        for p in model.heads.mask_head.parameters():
            p.requires_grad = False
        logger.info("Frozen: mask_head (not training this phase)")
    elif args.face_weight == 0.0 and args.mask_weight > 0:
        for p in model.heads.face_head.parameters():
            p.requires_grad = False
        logger.info("Frozen: face_head (not training this phase)")

    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in params):,}")

    optimizer   = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 1

    # ── Resume ──
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from: {args.resume}  epoch={ckpt['epoch']}")
        else:
            model.load_state_dict(ckpt)
            logger.info(f"Resumed weights only from: {args.resume}")
    else:
        logger.info("Starting fresh.")

    stopper = EarlyStopping(patience=args.patience, logger=logger)
    best_face_recall = 0.0
    best_mask_acc    = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_fmet, tr_mmet = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_fmet, va_mmet = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()

        csv_log.write(epoch,  "train", tr_loss, tr_fmet, tr_mmet)
        csv_log.write(epoch,  "val",   va_loss, va_fmet, va_mmet)
        jsonl_log.write(epoch, "train", tr_loss, tr_fmet, tr_mmet)
        jsonl_log.write(epoch, "val",   va_loss, va_fmet, va_mmet)

        va_frecall = va_fmet["recall"]   if va_fmet else 0.0
        va_macc    = va_mmet["accuracy"] if va_mmet else 0.0
        tr_frecall = tr_fmet["recall"]   if tr_fmet else 0.0
        tr_macc    = tr_mmet["accuracy"] if tr_mmet else 0.0

        face_str = f"face_recall {tr_frecall:.3f}/{va_frecall:.3f}  " if args.face_weight > 0 else ""
        mask_str = f"mask_acc {tr_macc:.3f}/{va_macc:.3f}  "          if args.mask_weight > 0 else ""

        logger.info(
            f"[{epoch:02d}/{args.epochs}]  "
            f"loss {tr_loss:.4f}/{va_loss:.4f}  "
            f"{face_str}{mask_str}"
            f"({time.time()-t0:.1f}s)"
        )

        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info(f"\n  -- Val metrics @ epoch {epoch} --")
            if args.face_weight > 0:
                log_metrics(logger, va_fmet, "Face/Non-Face")
            if args.mask_weight > 0:
                log_metrics(logger, va_mmet, "Mask/Non-Mask")

        # ── Checkpoints ──
        saved = []

        if args.face_weight > 0 and va_frecall > best_face_recall:
            best_face_recall = va_frecall
            torch.save({
                "epoch": epoch, "embedding_dim": args.embedding_dim,
                "backbone": "mobilenet_v3_small",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_recall": va_frecall,
            }, os.path.join(args.save_dir, "best_face_head.pt"))
            saved.append(f"face_head (val_recall={va_frecall:.4f})")

        if args.mask_weight > 0 and va_macc > best_mask_acc:
            best_mask_acc = va_macc
            torch.save({
                "epoch": epoch, "embedding_dim": args.embedding_dim,
                "backbone": "mobilenet_v3_small",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_acc": va_macc,
            }, os.path.join(args.save_dir, "best_mask_head.pt"))
            saved.append(f"mask_head (val_acc={va_macc:.4f})")

        if saved:
            logger.info(f"  -> saved: {', '.join(saved)}")

        primary_metric = va_frecall if args.face_weight > 0 else va_macc
        if stopper.step(primary_metric):
            break

    # ── Post-training ONNX export ──
    logger.info("\nExporting ONNX files for production...")

    best_ckpt = "best_face_head.pt" if args.face_weight > 0 else "best_mask_head.pt"
    best_path = os.path.join(args.save_dir, best_ckpt)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    """
train_mobilenetv3.py

End-to-end fine-tuning of MobileNetV3 Small backbone with dual classification heads.


  Phase 1 — Face/Non-Face only:
     python train_mobilenetv3.py \
         --data_root   data/ \
         --save_dir    checkpoints/mv3_phase1 \
         --mask_weight 0.0

  Phase 2 — Mask/Non-Mask only:
     python train_mobilenetv3.py \
         --data_root   data/ \
         --save_dir    checkpoints/mv3_phase2 \
         --face_weight 0.0 \
         --resume      checkpoints/mv3_phase1/best_face_head.pt

  After training, exports ONNX files to save_dir/:
     backbone.onnx          <- backbone for production
     classifier_heads.onnx  <- heads ONNX
     best_face_head.pt      <- PyTorch checkpoint

  Merge into single ONNX (existing script):
     python merge_to_single_onnx.py \
         --backbone  checkpoints/mv3_phase2/backbone.onnx \
         --heads     checkpoints/mv3_phase2/classifier_heads.onnx \
         --output    classifier_full_mv3.onnx --verify --benchmark

  Benchmark (existing script):
     python export_and_infer.py --mode benchmark \
         --fused_onnx classifier_full_mv3.onnx \
         --img_dir data/val/face --runs 200
"""

import argparse
import csv
import json
import logging
import os
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from mobilenetv3_model import (MobileNetV3SmallBackbone,
                               DualHeadClassifier, EndToEndClassifier)
from dataset import build_dataloaders


# --------------------------------------------------------------------------- #
#  Logger setup
# --------------------------------------------------------------------------- #

def setup_logger(save_dir: str) -> logging.Logger:
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(os.path.join(save_dir, "train.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# --------------------------------------------------------------------------- #
#  CSV logger
# --------------------------------------------------------------------------- #

class CSVLogger:
    FIELDS = [
        "epoch", "split", "loss",
        "face_acc", "face_precision", "face_recall", "face_f1",
        "face_fpr", "face_fnr", "face_auc",
        "mask_acc", "mask_precision", "mask_recall", "mask_f1",
        "mask_fpr", "mask_fnr", "mask_auc",
    ]

    def __init__(self, save_dir: str):
        self.path = os.path.join(save_dir, "metrics.csv")
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def _extract(self, metrics, prefix):
        if metrics is None:
            return {f"{prefix}_{k}": None
                    for k in ["acc","precision","recall","f1","fpr","fnr","auc"]}
        return {
            f"{prefix}_acc"       : round(metrics["accuracy"],  4),
            f"{prefix}_precision" : round(metrics["precision"], 4),
            f"{prefix}_recall"    : round(metrics["recall"],    4),
            f"{prefix}_f1"        : round(metrics["f1"],        4),
            f"{prefix}_fpr"       : round(metrics["fpr"],       4),
            f"{prefix}_fnr"       : round(metrics["fnr"],       4),
            f"{prefix}_auc"       : round(metrics["roc_auc"], 4) if metrics["roc_auc"] else None,
        }

    def write(self, epoch, split, loss, face_metrics, mask_metrics):
        row = {"epoch": epoch, "split": split, "loss": round(loss, 6)}
        row.update(self._extract(face_metrics, "face"))
        row.update(self._extract(mask_metrics, "mask"))
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(row)


# --------------------------------------------------------------------------- #
#  JSONL logger
# --------------------------------------------------------------------------- #

class JSONLLogger:
    def __init__(self, save_dir: str):
        self.path = os.path.join(save_dir, "metrics.jsonl")
        open(self.path, "w").close()

    def write(self, epoch, split, loss, face_metrics, mask_metrics):
        record = {
            "epoch": epoch, "split": split, "loss": round(loss, 6),
            "face_metrics": face_metrics, "mask_metrics": mask_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- #
#  Early stopping
# --------------------------------------------------------------------------- #

class EarlyStopping:
    def __init__(self, patience: int, logger: logging.Logger):
        self.patience = patience
        self.logger   = logger
        self.counter  = 0
        self.best     = None

    def step(self, metric: float) -> bool:
        if self.patience <= 0:
            return False
        if self.best is None or metric > self.best:
            self.best    = metric
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(
                f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs"
            )
            if self.counter >= self.patience:
                self.logger.info(f"  [EarlyStopping] Stopping early")
                return True
        return False


# --------------------------------------------------------------------------- #
#  Loss
# --------------------------------------------------------------------------- #



class DualHeadLoss(nn.Module):
    def __init__(self, face_weight=1.0, mask_weight=1.0,
                 face_class_weights=None):
        super().__init__()
        self.face_w = face_weight
        self.mask_w = mask_weight

        # Face head: weight non-face (class 0) higher to fix FPR
        # [3.0, 1.0] compensates for ~3:1 face/non-face imbalance
        if face_class_weights is not None:
            self.face_ce = nn.CrossEntropyLoss(
                weight=torch.tensor(face_class_weights),
                ignore_index=-1
            )
        else:
            self.face_ce = nn.CrossEntropyLoss(ignore_index=-1)

        # Mask head: balanced dataset, no class weights needed
        self.mask_ce = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, face_logits, mask_logits, face_labels, mask_labels):
        device = face_logits.device
        self.face_ce.weight = self.face_ce.weight.to(device) if self.face_ce.weight is not None else None

        f = torch.tensor(0.0, device=device)
        m = torch.tensor(0.0, device=device)
        if self.face_w > 0 and (face_labels != -1).any():
            f = self.face_ce(face_logits, face_labels)
        if self.mask_w > 0 and (mask_labels != -1).any():
            m = self.mask_ce(mask_logits, mask_labels)
        loss = self.face_w * f + self.mask_w * m
        return loss, float(f.detach()), float(m.detach())

# class DualHeadLoss(nn.Module):
#     def __init__(self, face_weight=1.0, mask_weight=1.0):
#         super().__init__()
#         self.face_w = face_weight
#         self.mask_w = mask_weight
#         self.ce = nn.CrossEntropyLoss(ignore_index=-1)

#     def forward(self, face_logits, mask_logits, face_labels, mask_labels):
#         device = face_logits.device
#         f = torch.tensor(0.0, device=device)
#         m = torch.tensor(0.0, device=device)
#         if self.face_w > 0 and (face_labels != -1).any():
#             f = self.ce(face_logits, face_labels)
#         if self.mask_w > 0 and (mask_labels != -1).any():
#             m = self.ce(mask_logits, mask_labels)
#         loss = self.face_w * f + self.mask_w * m
#         return loss, float(f.detach()), float(m.detach())


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #

def compute_metrics(all_preds, all_labels, all_probs, ignore=-1):
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, confusion_matrix
    )
    preds  = np.array(all_preds)
    labels = np.array(all_labels)
    probs  = np.array(all_probs)

    valid  = labels != ignore
    preds, labels, probs = preds[valid], labels[valid], probs[valid]

    if len(labels) == 0:
        return None

    has_both = len(np.unique(labels)) == 2
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy"  : accuracy_score(labels, preds),
        "precision" : precision_score(labels, preds, zero_division=0),
        "recall"    : recall_score(labels, preds, zero_division=0),
        "f1"        : f1_score(labels, preds, zero_division=0),
        "fpr"       : fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr"       : fn / (fn + tp) if (fn + tp) > 0 else 0.0,
        "roc_auc"   : roc_auc_score(labels, probs) if has_both else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def log_metrics(logger, metrics, head_name):
    if metrics is None:
        logger.info(f"  [{head_name}] Not enough data.")
        return
    cm = metrics["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    auc = f"{metrics['roc_auc']:.4f}" if metrics["roc_auc"] else "N/A"
    logger.info(f"  [{head_name}]")
    logger.info(f"    Accuracy  : {metrics['accuracy']:.4f}")
    logger.info(f"    Precision : {metrics['precision']:.4f}")
    logger.info(f"    Recall    : {metrics['recall']:.4f}")
    logger.info(f"    F1        : {metrics['f1']:.4f}")
    logger.info(f"    FPR       : {metrics['fpr']:.4f}")
    logger.info(f"    FNR       : {metrics['fnr']:.4f}")
    logger.info(f"    ROC-AUC   : {auc}")
    logger.info(f"    CM: TN={tn} FP={fp} FN={fn} TP={tp}")


# --------------------------------------------------------------------------- #
#  One epoch
# --------------------------------------------------------------------------- #

def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()
    total_loss = n = 0
    face_preds, face_labels, face_probs = [], [], []
    mask_preds, mask_labels, mask_probs = [], [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pbar = tqdm(loader, desc=f"  {'train' if train else 'val'}",
                    leave=False, unit="batch", dynamic_ncols=True)
        for imgs, face_lbl, mask_lbl in pbar:
            imgs     = imgs.to(device)
            face_lbl = face_lbl.to(device)
            mask_lbl = mask_lbl.to(device)

            face_logits, mask_logits = model(imgs)
            loss, _, _ = criterion(face_logits, mask_logits, face_lbl, mask_lbl)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item(); n += 1
            pbar.set_postfix(loss=f"{total_loss/n:.4f}")

            face_preds  += face_logits.argmax(1).cpu().tolist()
            face_labels += face_lbl.cpu().tolist()
            face_probs  += torch.softmax(face_logits, 1)[:, 1].cpu().tolist()
            mask_preds  += mask_logits.argmax(1).cpu().tolist()
            mask_labels += mask_lbl.cpu().tolist()
            mask_probs  += torch.softmax(mask_logits, 1)[:, 1].cpu().tolist()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return (total_loss / n,
            compute_metrics(face_preds, face_labels, face_probs),
            compute_metrics(mask_preds, mask_labels, mask_probs))


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",     required=True)
    p.add_argument("--save_dir",      default="checkpoints")
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--epochs",        type=int,   default=50)
    p.add_argument("--patience",      type=int,   default=7)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--face_weight",   type=float, default=1.0)
    p.add_argument("--mask_weight",   type=float, default=1.0)
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--no_pretrained", action="store_true",
                   help="Disable ImageNet pretrained weights")
    p.add_argument("--resume",        default=None)
    return p.parse_args()


def main():
    args      = parse_args()
    logger    = setup_logger(args.save_dir)
    csv_log   = CSVLogger(args.save_dir)
    jsonl_log = JSONLLogger(args.save_dir)

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = torch.cuda.is_available()

    # Phase detection
    if args.face_weight > 0 and args.mask_weight == 0.0:
        phase = "Phase 1 — Face/Non-Face only"
    elif args.face_weight == 0.0 and args.mask_weight > 0:
        phase = "Phase 2 — Mask/Non-Mask only"
    else:
        phase = "Full — both heads"

    # ── Build model ──
    backbone = MobileNetV3SmallBackbone(
        embedding_dim=args.embedding_dim,
        pretrained=not args.no_pretrained,
    )
    heads = DualHeadClassifier(embedding_dim=args.embedding_dim)
    model = EndToEndClassifier(backbone, heads).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("=" * 60)
    logger.info(f"  Run started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"  Backbone        : MobileNetV3 Small")
    logger.info(f"  Phase           : {phase}")
    logger.info(f"  Device          : {device}")
    logger.info(f"  Total params    : {total_params:,}")
    logger.info(f"  Embedding dim   : {args.embedding_dim}")
    logger.info(f"  face_weight     : {args.face_weight}")
    logger.info(f"  mask_weight     : {args.mask_weight}")
    logger.info(f"  Epochs          : {args.epochs}")
    logger.info(f"  Patience        : {args.patience if args.patience > 0 else 'disabled'}")
    logger.info(f"  Batch size      : {args.batch_size}")
    logger.info(f"  LR              : {args.lr}")
    logger.info("=" * 60)

    train_loader, val_loader = build_dataloaders(
        args.data_root, args.batch_size, args.num_workers,
        pin_memory=pin_memory
    )

    # criterion = DualHeadLoss(args.face_weight, args.mask_weight)
    criterion = DualHeadLoss(
            args.face_weight, args.mask_weight,
            face_class_weights=[3.0, 1.0]
        )

    # ── Optimizer — freeze inactive head for single-phase training ──
    if args.face_weight > 0 and args.mask_weight == 0.0:
        for p in model.heads.mask_head.parameters():
            p.requires_grad = False
        logger.info("Frozen: mask_head (not training this phase)")
    elif args.face_weight == 0.0 and args.mask_weight > 0:
        for p in model.heads.face_head.parameters():
            p.requires_grad = False
        logger.info("Frozen: face_head (not training this phase)")

    params = [p for p in model.parameters() if p.requires_grad]
    logger.info(f"Trainable params: {sum(p.numel() for p in params):,}")

    optimizer   = AdamW(params, lr=args.lr, weight_decay=1e-4)
    scheduler   = CosineAnnealingLR(optimizer, T_max=args.epochs)
    start_epoch = 1

    # ── Resume ──
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)

        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(f"Resumed from: {args.resume}  epoch={ckpt['epoch']}")
        else:
            model.load_state_dict(ckpt)
            logger.info(f"Resumed weights only from: {args.resume}")
    else:
        logger.info("Starting fresh.")

    stopper = EarlyStopping(patience=args.patience, logger=logger)
    best_face_recall = 0.0
    best_mask_acc    = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_fmet, tr_mmet = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True)
        va_loss, va_fmet, va_mmet = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()

        csv_log.write(epoch,  "train", tr_loss, tr_fmet, tr_mmet)
        csv_log.write(epoch,  "val",   va_loss, va_fmet, va_mmet)
        jsonl_log.write(epoch, "train", tr_loss, tr_fmet, tr_mmet)
        jsonl_log.write(epoch, "val",   va_loss, va_fmet, va_mmet)

        va_frecall = va_fmet["recall"]   if va_fmet else 0.0
        va_macc    = va_mmet["accuracy"] if va_mmet else 0.0
        tr_frecall = tr_fmet["recall"]   if tr_fmet else 0.0
        tr_macc    = tr_mmet["accuracy"] if tr_mmet else 0.0

        face_str = f"face_recall {tr_frecall:.3f}/{va_frecall:.3f}  " if args.face_weight > 0 else ""
        mask_str = f"mask_acc {tr_macc:.3f}/{va_macc:.3f}  "          if args.mask_weight > 0 else ""

        logger.info(
            f"[{epoch:02d}/{args.epochs}]  "
            f"loss {tr_loss:.4f}/{va_loss:.4f}  "
            f"{face_str}{mask_str}"
            f"({time.time()-t0:.1f}s)"
        )

        if epoch % 5 == 0 or epoch == args.epochs:
            logger.info(f"\n  -- Val metrics @ epoch {epoch} --")
            if args.face_weight > 0:
                log_metrics(logger, va_fmet, "Face/Non-Face")
            if args.mask_weight > 0:
                log_metrics(logger, va_mmet, "Mask/Non-Mask")

        # ── Checkpoints ──
        saved = []

        if args.face_weight > 0 and va_frecall > best_face_recall:
            best_face_recall = va_frecall
            torch.save({
                "epoch": epoch, "embedding_dim": args.embedding_dim,
                "backbone": "mobilenet_v3_small",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_recall": va_frecall,
            }, os.path.join(args.save_dir, "best_face_head.pt"))
            saved.append(f"face_head (val_recall={va_frecall:.4f})")

        if args.mask_weight > 0 and va_macc > best_mask_acc:
            best_mask_acc = va_macc
            torch.save({
                "epoch": epoch, "embedding_dim": args.embedding_dim,
                "backbone": "mobilenet_v3_small",
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "val_acc": va_macc,
            }, os.path.join(args.save_dir, "best_mask_head.pt"))
            saved.append(f"mask_head (val_acc={va_macc:.4f})")

        if saved:
            logger.info(f"  -> saved: {', '.join(saved)}")

        primary_metric = va_frecall if args.face_weight > 0 else va_macc
        if stopper.step(primary_metric):
            break

    # ── Post-training ONNX export ──
    logger.info("\nExporting ONNX files for production...")

    best_ckpt = "best_face_head.pt" if args.face_weight > 0 else "best_mask_head.pt"
    best_path = os.path.join(args.save_dir, best_ckpt)
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state"])

    model.cpu().eval()
    model.export_backbone_onnx(os.path.join(args.save_dir, "backbone.onnx"))
    model.export_heads_onnx(os.path.join(args.save_dir, "classifier_heads.onnx"))

    logger.info(f"\nProduction files:")
    logger.info(f"  {args.save_dir}/backbone.onnx          <- use as --onnx_path")
    logger.info(f"  {args.save_dir}/classifier_heads.onnx  <- use as --heads_onnx")
    logger.info(f"\nMerge into single ONNX:")
    logger.info(f"  python merge_to_single_onnx.py "
                f"--backbone {args.save_dir}/backbone.onnx "
                f"--heads {args.save_dir}/classifier_heads.onnx")

    logger.info("\nDone.")
    if args.face_weight > 0:
        logger.info(f"  Best face head val_recall : {best_face_recall:.4f}")
    if args.mask_weight > 0:
        logger.info(f"  Best mask head val_acc    : {best_mask_acc:.4f}")


if __name__ == "__main__":
    main()

    logger.info("\nDone.")
    if args.face_weight > 0:
        logger.info(f"  Best face head val_recall : {best_face_recall:.4f}")
    if args.mask_weight > 0:
        logger.info(f"  Best mask head val_acc    : {best_mask_acc:.4f}")


if __name__ == "__main__":
    main()