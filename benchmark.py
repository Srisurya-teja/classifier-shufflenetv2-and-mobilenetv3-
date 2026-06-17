"""
benchmark.py

Benchmark the fused ONNX classifier on real images.

Usage:
    python benchmark.py \
        --fused_onnx checkpoints/shuffle_phase2_weighted/classifier_full.onnx \
        --img_dir    /data/val/face \
        --runs       200
"""

import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort


FACE_LABELS = {0: "non_face", 1: "face"}
MASK_LABELS = {0: "no_mask",  1: "mask"}




def preprocess(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.dnn.blobFromImage(
        img_bgr, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )



def softmax(v: np.ndarray) -> np.ndarray:
    e = np.exp(v - v.max())
    return e / e.sum()



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



class FusedClassifierRuntime:
    def __init__(self, fused_onnx: str):
        self.sess = build_session(fused_onnx)
        self.inp  = self.sess.get_inputs()[0].name

        print(f"[runtime] Provider : {self.sess.get_providers()[0]}")
        print(f"[runtime] Model    : {fused_onnx}")
        print(f"[runtime] Input    : {self.inp}  {self.sess.get_inputs()[0].shape}")

    def predict(self, img_bgr: np.ndarray):
        x = preprocess(img_bgr)
        outputs = self.sess.run(None, {self.inp: x})

        fp = softmax(outputs[0][0]); fi = int(fp.argmax())
        mp = softmax(outputs[1][0]); mi = int(mp.argmax())

        return FACE_LABELS[fi], float(fp[fi]), MASK_LABELS[mi], float(mp[mi])




def benchmark(rt, img_dir: str, runs: int = 200):
    exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
             if os.path.splitext(f)[1].lower() in exts][:runs]

    if not paths:
        print("[bench] No images found.")
        return

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

    print(f"\n===== Inference Benchmark =====")
    print(f"  Images  : {len(lats)}")
    print(f"  Mean    : {lats.mean():.2f} ms")
    print(f"  Median  : {np.median(lats):.2f} ms")
    print(f"  P95     : {np.percentile(lats, 95):.2f} ms")
    print(f"  P99     : {np.percentile(lats, 99):.2f} ms")
    print(f"\n  2ms budget (P95): {'PASS ✓' if np.percentile(lats, 95) <= 2.0 else 'FAIL ✗'}")

    print(f"\n===== Sample Predictions =====")
    for p in paths[:5]:
        fl, fc, ml, mc = rt.predict(cv2.imread(p))
        print(f"  {os.path.basename(p):30s}  face={fl}({fc:.2f})  mask={ml}({mc:.2f})")




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--fused_onnx", required=True, help="classifier_full.onnx")
    p.add_argument("--img_dir",    required=True, help="Directory of 112x112 face crops")
    p.add_argument("--runs",       type=int, default=200)
    args = p.parse_args()

    rt = FusedClassifierRuntime(args.fused_onnx)
    benchmark(rt, args.img_dir, args.runs)