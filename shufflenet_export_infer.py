"""
export_and_infer.py

Export:
    python export_and_infer.py --mode export \
        --onnx_path  checkpoints/phase2/backbone.onnx \
        --face_ckpt  checkpoints/phase1/best_face_head.pt \
        --mask_ckpt  checkpoints/phase2/best_mask_head.pt \
        --output     classifier_heads.onnx

Optimize (save ORT-optimized graph — run once):
    python export_and_infer.py --mode optimize \
        --fused_onnx classifier_full.onnx \
        --output     classifier_full_opt.onnx

Benchmark (chained — two ONNX files):
    python export_and_infer.py --mode benchmark \
        --onnx_path  checkpoints/phase2/backbone.onnx \
        --heads_onnx classifier_heads.onnx \
        --img_dir    data/val/face \
        --runs       200

Benchmark (fused — single ONNX file):
    python export_and_infer.py --mode benchmark \
        --fused_onnx classifier_full.onnx \
        --img_dir    data/val/face \
        --runs       200
"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch

from shufflenet_model import DualHeadClassifier


FACE_LABELS = {0: "non_face", 1: "face"}
MASK_LABELS = {0: "no_mask",  1: "mask"}

_FACE_KEYS = list(FACE_LABELS.keys())
_MASK_KEYS = list(MASK_LABELS.keys())


# --------------------------------------------------------------------------- #
#  Preprocessing — single C++ call via OpenCV
# --------------------------------------------------------------------------- #

def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    """
    Input : 112x112x3 BGR uint8
    Output: 1x3x112x112 float32  (RGB, normalized to [-1, 1])

    cv2.dnn.blobFromImage does in one C++ call:
      - subtract mean (127.5)
      - multiply by scalefactor (1/127.5)
      - BGR → RGB (swapRB=True)
      - HWC → NCHW
      - add batch dim
    Replaces 5 separate numpy operations.
    """
    return cv2.dnn.blobFromImage(
        img_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )


# --------------------------------------------------------------------------- #
#  Softmax — defined once, reused everywhere
# --------------------------------------------------------------------------- #

def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()


# --------------------------------------------------------------------------- #
#  Helper — extract heads weights from EndToEndClassifier checkpoint
# --------------------------------------------------------------------------- #

def extract_heads_weights(ckpt_path: str) -> tuple:
    """
    Extracts only the DualHeadClassifier state_dict from a checkpoint
    that contains the full EndToEndClassifier (backbone + heads).

    Returns (heads_state_dict, embedding_dim or None).

    Handles three checkpoint formats:
      1. EndToEnd checkpoint: keys like 'heads.face_head.0.weight'
         → strips 'heads.' prefix, discards backbone keys
      2. Heads-only checkpoint: keys like 'face_head.0.weight'
         → returns as-is
      3. Weights-only (bare state_dict, no 'model_state' wrapper)
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state     = ckpt["model_state"]
        embed_dim = ckpt.get("embedding_dim", None)
        epoch     = ckpt.get("epoch", "?")

        # EndToEndClassifier checkpoint — has 'heads.' and 'backbone.' prefixes
        if any(k.startswith("heads.") for k in state):
            heads_state = {k.replace("heads.", ""): v
                           for k, v in state.items()
                           if k.startswith("heads.")}
            print(f"  [ckpt] EndToEnd checkpoint — epoch {epoch}"
                  f"  embedding_dim={embed_dim or 'not stored'}"
                  f"  (extracted {len(heads_state)} head keys)")
            return heads_state, embed_dim

        # Already heads-only (no 'heads.' prefix, no 'backbone.' keys)
        print(f"  [ckpt] Heads-only checkpoint — epoch {epoch}"
              f"  embedding_dim={embed_dim or 'not stored'}")
        return state, embed_dim

    # Bare state_dict (no wrapper)
    print(f"  [ckpt] Weights-only checkpoint")
    return ckpt, None


# --------------------------------------------------------------------------- #
#  ORT session builder — tuned for low-latency small models
# --------------------------------------------------------------------------- #

def build_session(onnx_path: str) -> ort.InferenceSession:
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level       = 3
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.enable_cpu_mem_arena     = True
    sess_options.enable_mem_pattern       = True

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    return ort.InferenceSession(onnx_path, sess_options=sess_options,
                                providers=providers)


# --------------------------------------------------------------------------- #
#  Save ORT-optimized graph (run once, reuse forever)
# --------------------------------------------------------------------------- #

def optimize_graph(input_onnx: str, output_onnx: str):
    """
    Loads the model with ORT_ENABLE_ALL, saves the optimized graph.
    The saved model loads faster and skips re-optimization on every run.
    """
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level       = 3
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.optimized_model_filepath = output_onnx

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    ort.InferenceSession(input_onnx, sess_options=sess_options,
                         providers=providers)

    orig_mb = os.path.getsize(input_onnx) / 1e6
    opt_mb  = os.path.getsize(output_onnx) / 1e6
    print(f"[optimize] {input_onnx} ({orig_mb:.2f} MB)")
    print(f"[optimize] → {output_onnx} ({opt_mb:.2f} MB)")
    print(f"[optimize] Benchmark with:")
    print(f"  python export_and_infer.py --mode benchmark "
          f"--fused_onnx {output_onnx} --img_dir data/val/face --runs 200")


# --------------------------------------------------------------------------- #
#  Export
# --------------------------------------------------------------------------- #

def export(backbone_onnx: str, face_ckpt: str, mask_ckpt: str,
           output_path: str, embedding_dim: int):

    heads = DualHeadClassifier(embedding_dim)

    base_state, ckpt_dim = extract_heads_weights(face_ckpt)

    # Sanity check: warn if checkpoint was trained with a different dim
    if ckpt_dim is not None and ckpt_dim != embedding_dim:
        print(f"  [WARNING] Checkpoint embedding_dim={ckpt_dim} "
              f"vs requested --embedding_dim={embedding_dim}")

    heads.load_state_dict(base_state)
    print(f"[export] Loaded face head from : {face_ckpt}")

    if mask_ckpt and os.path.exists(mask_ckpt):
        mask_state, _ = extract_heads_weights(mask_ckpt)
        for k, v in mask_state.items():
            if k.startswith("mask_head."):
                base_state[k] = v
        heads.load_state_dict(base_state)
        print(f"[export] Merged mask head from : {mask_ckpt}")
    else:
        print(f"[export] No mask checkpoint — mask head is untrained")

    heads.eval()

    dummy = torch.randn(1, embedding_dim)
    torch.onnx.export(
        heads, dummy, output_path,
        input_names=["embedding"],
        output_names=["face_logits", "mask_logits"],
        dynamic_axes={"embedding": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    mb = os.path.getsize(output_path) / 1e6
    print(f"[export] Saved → {output_path}  ({mb:.2f} MB)")
    print(f"[export] Embedding dim : {embedding_dim}")
    print(f"[export] Use alongside backbone : {backbone_onnx}")


# --------------------------------------------------------------------------- #
#  Runtime — chained (two ONNX files)
# --------------------------------------------------------------------------- #

class ClassifierRuntime:
    """
    Chained runtime — backbone ONNX + heads ONNX as two separate sessions.
    """

    def __init__(self, backbone_onnx: str, heads_onnx: str):
        self.backbone = build_session(backbone_onnx)
        self.heads    = build_session(heads_onnx)

        self.bb_in  = self.backbone.get_inputs()[0].name
        self.bb_out = self.backbone.get_outputs()[0].name
        self.h_in   = self.heads.get_inputs()[0].name

        print(f"[runtime] Mode     : chained (2 ONNX files)")
        print(f"[runtime] Provider : {self.backbone.get_providers()[0]}")
        print(f"[runtime] Backbone : {backbone_onnx}")
        print(f"[runtime] Heads    : {heads_onnx}")

    def predict(self, img_bgr: np.ndarray):
        x   = preprocess(img_bgr)
        emb = self.backbone.run([self.bb_out], {self.bb_in: x})[0]

        face_logits, mask_logits = self.heads.run(
            ["face_logits", "mask_logits"], {self.h_in: emb}
        )

        fp = softmax(face_logits[0]); fi = int(fp.argmax())
        mp = softmax(mask_logits[0]); mi = int(mp.argmax())

        return FACE_LABELS[fi], float(fp[fi]), MASK_LABELS[mi], float(mp[mi])


# --------------------------------------------------------------------------- #
#  Runtime — fused (single ONNX file)
# --------------------------------------------------------------------------- #

class FusedClassifierRuntime:
    """
    Fused runtime — single ONNX, simple sess.run(), cv2 preprocessing.
    """

    def __init__(self, fused_onnx: str):
        self.sess = build_session(fused_onnx)
        self.inp  = self.sess.get_inputs()[0].name

        print(f"[runtime] Mode     : fused (1 ONNX file)")
        print(f"[runtime] Provider : {self.sess.get_providers()[0]}")
        print(f"[runtime] Model    : {fused_onnx}")
        print(f"[runtime] Input    : {self.inp}  {self.sess.get_inputs()[0].shape}")

    def predict(self, img_bgr: np.ndarray):
        x = preprocess(img_bgr)
        outputs = self.sess.run(None, {self.inp: x})

        face_logits = outputs[0]
        mask_logits = outputs[1]

        fp = softmax(face_logits[0]); fi = int(fp.argmax())
        mp = softmax(mask_logits[0]); mi = int(mp.argmax())

        return FACE_LABELS[fi], float(fp[fi]), MASK_LABELS[mi], float(mp[mi])


# --------------------------------------------------------------------------- #
#  Benchmark
# --------------------------------------------------------------------------- #

def benchmark(rt, img_dir: str, runs: int = 200, mode_label: str = ""):
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in exts][:runs]

    if not paths:
        print("[bench] No images found.")
        return

    # Extended warm-up — stabilizes CPU caches and frequency scaling
    print(f"[bench] Warming up (20 runs)...")
    for p in (paths[:5] * 4):
        rt.predict(cv2.imread(p))

    lats = []
    for p in paths:
        img = cv2.imread(p)
        t0  = time.perf_counter()
        rt.predict(img)
        lats.append((time.perf_counter() - t0) * 1000)

    lats = np.array(lats)
    p95  = np.percentile(lats, 95)

    label = f" ({mode_label})" if mode_label else ""
    print(f"\n===== Inference Benchmark{label} =====")
    print(f"  Images  : {len(lats)}")
    print(f"  Mean    : {lats.mean():.2f} ms")
    print(f"  Median  : {np.median(lats):.2f} ms")
    print(f"  P95     : {p95:.2f} ms")
    print(f"  P99     : {np.percentile(lats, 99):.2f} ms")
    print(f"\n  5ms budget (P95): {'PASS ✓' if p95 <= 5.0 else 'FAIL ✗'}")

    print(f"\n===== Sample Predictions =====")
    for p in paths[:5]:
        fl, fc, ml, mc = rt.predict(cv2.imread(p))
        print(f"  {os.path.basename(p):30s}  face={fl}({fc:.2f})  mask={ml}({mc:.2f})")


# --------------------------------------------------------------------------- #
#  Entry point
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",       required=True,
                   choices=["export", "optimize", "benchmark"])

    # Export args
    p.add_argument("--onnx_path",      default=None,  help="backbone ONNX")
    p.add_argument("--face_ckpt",      default=None,  help="best_face_head.pt  (export)")
    p.add_argument("--mask_ckpt",      default=None,  help="best_mask_head.pt  (export)")
    p.add_argument("--embedding_dim",  type=int, default=None,
                   help="Auto-detected from checkpoint if omitted. "
                        "128 for ShuffleNetV2 x0.5.")
    p.add_argument("--output",         default="classifier_heads.onnx")

    # Benchmark args — chained
    p.add_argument("--heads_onnx", default=None,
                   help="classifier_heads.onnx (chained benchmark)")

    # Benchmark / Optimize args — fused
    p.add_argument("--fused_onnx", default=None,
                   help="classifier_full.onnx  (fused benchmark / optimize)")

    p.add_argument("--img_dir",    default=None)
    p.add_argument("--runs",       type=int, default=200)
    return p.parse_args()


def _resolve_embedding_dim(args) -> int:
    """Resolve embedding_dim from CLI > checkpoint > fallback 128."""
    if args.embedding_dim is not None:
        return args.embedding_dim

    # Try to read from checkpoint
    if args.face_ckpt and os.path.exists(args.face_ckpt):
        ckpt = torch.load(args.face_ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "embedding_dim" in ckpt:
            dim = ckpt["embedding_dim"]
            print(f"[export] Auto-detected embedding_dim={dim} from checkpoint")
            return dim

    print(f"[export] Using default embedding_dim=128 "
          f"(pass --embedding_dim to override)")
    return 128


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "export":
        assert args.face_ckpt,  "--face_ckpt required for export"
        assert args.onnx_path,  "--onnx_path required for export"
        embed_dim = _resolve_embedding_dim(args)
        export(args.onnx_path, args.face_ckpt, args.mask_ckpt,
               args.output, embed_dim)

    elif args.mode == "optimize":
        assert args.fused_onnx, "--fused_onnx required for optimize"
        optimize_graph(args.fused_onnx, args.output)

    else:  # benchmark
        assert args.img_dir, "--img_dir required for benchmark"

        if args.fused_onnx:
            rt = FusedClassifierRuntime(args.fused_onnx)
            benchmark(rt, args.img_dir, args.runs, mode_label="fused")

        elif args.onnx_path and args.heads_onnx:
            rt = ClassifierRuntime(args.onnx_path, args.heads_onnx)
            benchmark(rt, args.img_dir, args.runs, mode_label="chained")

        else:
            print("For benchmark provide either:")
            print("  --fused_onnx classifier_full.onnx")
            print("  OR")
            print("  --onnx_path backbone.onnx --heads_onnx classifier_heads.onnx")