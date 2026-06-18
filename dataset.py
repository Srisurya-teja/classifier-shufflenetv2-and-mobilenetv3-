"""
dataset.py

Loads 112x112 BGR crops produced by RetinaFace norm_crop() and applies
the exact same preprocessing that InsightFace's ArcFaceONNX uses internally.

No PIL, no torchvision Normalize — just numpy ops matching the backbone's
training preprocessing.

Folder layout:
    data/
      train/
        face/        <- RetinaFace-cropped face images (112x112 BGR)
        non-face/    <- non-face data run through onnx_inference.py
        mask/        <- masked face crops (from Yotta)
        non-mask/    <- unmasked face crops
      val/
        ...same...
"""

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# --------------------------------------------------------------------------- #
#  Preprocessing — matches InsightFace ArcFaceONNX exactly
# --------------------------------------------------------------------------- #

def retina_preprocess(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Input : 112x112x3 BGR uint8 numpy array (direct RetinaFace norm_crop output)
    Output: 3x112x112 float32 tensor, range [-1, 1]
    """
    img = img_bgr.astype(np.float32)
    img = (img - 127.5) / 127.5           # [0,255] → [-1, 1]
    img = img[:, :, ::-1]                 # BGR → RGB
    img = np.transpose(img, (2, 0, 1))    # HWC → CHW
    return torch.from_numpy(img.copy())   # copy() needed after ::-1 flip


def retina_preprocess_aug(img_bgr: np.ndarray) -> torch.Tensor:
    """
    Same as retina_preprocess but with light augmentation for training.
    All ops stay in numpy/cv2 to avoid PIL channel-order confusion.
    """
    # Random horizontal flip
    if np.random.rand() < 0.5:
        img_bgr = img_bgr[:, ::-1, :]

    # Random brightness / contrast jitter
    alpha = 1.0 + np.random.uniform(-0.3, 0.3)   # contrast
    beta  = np.random.uniform(-30, 30)            # brightness
    img_bgr = np.clip(img_bgr.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    # Small rotation (±10°)
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((56, 56), angle, 1.0)
    img_bgr = cv2.warpAffine(img_bgr, M, (112, 112),
                              borderMode=cv2.BORDER_REFLECT_101)

    return retina_preprocess(img_bgr)


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #

class FaceMaskDataset(Dataset):
    """
    face_label : 1 = face,    0 = non-face
    mask_label : 1 = mask,    0 = no-mask,   -1 = unknown (loss ignored)

    Non-face images get mask_label = -1 because they carry no mask signal.
    The loss uses ignore_index=-1 so the mask head gets zero gradient
    from those samples.
    """

    DIRS = {
        # folder      face_label  mask_label
        "face":     (1, -1),   # face, mask status unknown
        "non-face": (0, -1),   # non-face, mask irrelevant
        "mask":     (1,  1),   # face + mask
        "non-mask":  (1,  0),   # face + no mask
    }

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str, augment: bool = False):
        self.augment  = augment
        self.samples  = []   # (path, face_label, mask_label)

        root = Path(root)
        for folder, (fl, ml) in self.DIRS.items():
            p = root / folder
            if not p.exists():
                print(f"[dataset] Warning: {p} not found, skipping.")
                continue
            for f in p.iterdir():
                if f.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((str(f), fl, ml))

        self._log(root)

    def _log(self, root):
        from collections import Counter
        print(f"[dataset] {root}  total={len(self.samples):,}")
        counts = Counter(Path(p).parent.name for p, _, _ in self.samples)
        for folder, n in counts.items():
            print(f"  {folder:12s}: {n:>8,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, face_lbl, mask_lbl = self.samples[idx]

        img_bgr = cv2.imread(path)          # reads as BGR uint8 — same as RetinaFace
        if img_bgr is None:
            # Fallback: return a blank image rather than crashing the loader
            img_bgr = np.zeros((112, 112, 3), dtype=np.uint8)

        if img_bgr.shape[:2] != (112, 112):
            img_bgr = cv2.resize(img_bgr, (112, 112))

        if self.augment:
            tensor = retina_preprocess_aug(img_bgr)
        else:
            tensor = retina_preprocess(img_bgr)

        return (
            tensor,
            torch.tensor(face_lbl, dtype=torch.long),
            torch.tensor(mask_lbl, dtype=torch.long),
        )


def build_dataloaders(data_root: str, batch_size: int = 256, num_workers: int = 4,
                      pin_memory: bool = True):
    import shutil
    shm_free_gb = shutil.disk_usage("/dev/shm").free / 1e9
    if num_workers > 0 and shm_free_gb < 1.0:
        print(f"[dataset] WARNING: /dev/shm only {shm_free_gb:.2f} GB free — "
              f"falling back to num_workers=0. Run with --shm-size=4g for faster loading.")
        num_workers = 0

    train_ds = FaceMaskDataset(os.path.join(data_root, "train"), augment=True)
    val_ds   = FaceMaskDataset(os.path.join(data_root, "val"),   augment=False)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader
