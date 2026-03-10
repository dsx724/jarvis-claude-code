#!/usr/bin/env python3
"""Benchmark ONNX model inference: CPU vs OpenVINO CPU vs OpenVINO GPU(s).

Auto-detects available OpenVINO GPU devices and benchmarks all of them.
Use --gpu GPU.1 to benchmark only a specific GPU device.

Tests all models used by Jarvis with realistic input sizes:
  - melspectrogram:  1280 samples (80ms chunk, as fed by wake word loop)
  - embedding:       76x32x1 melspec window (~600ms of audio context)
  - hey_jarvis:      1x16x96 (16 embeddings = ~1.3s sliding window)
  - silero_vad:      512 samples (32ms chunk, as fed by VAD loop)
  - piper_tts:       ~130 phoneme IDs (typical spoken sentence)

All models are warmed up before timing to measure steady-state performance.
"""

import argparse
import os
import time

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WARMUP_ITERS = 10
BENCH_ITERS = 50


def try_load(path, providers):
    """Try to load a model with given providers, return session or None."""
    try:
        fd = os.dup(2)
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        try:
            sess = ort.InferenceSession(path, providers=providers)
        finally:
            os.dup2(fd, 2)
            os.close(fd)
        return sess
    except Exception:
        return None


def detect_gpu_devices(probe_model_path):
    """Detect available OpenVINO GPU devices by probing GPU.0, GPU.1, etc."""
    if "OpenVINOExecutionProvider" not in ort.get_available_providers():
        return []
    devices = []
    for i in range(8):
        dev = f"GPU.{i}"
        sess = try_load(probe_model_path, [
            ("OpenVINOExecutionProvider", {"device_type": dev}),
            "CPUExecutionProvider",
        ])
        if sess is None or "OpenVINOExecutionProvider" not in sess.get_providers():
            break
        devices.append(dev)
    return devices


def bench(sess, feed, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    """Run warmup + timed iterations, return mean ms per inference."""
    for _ in range(warmup):
        sess.run(None, feed)
    t0 = time.perf_counter()
    for _ in range(iters):
        sess.run(None, feed)
    return ((time.perf_counter() - t0) / iters) * 1000


def main():
    parser = argparse.ArgumentParser(description="Benchmark ONNX models on CPU vs GPU")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Specific GPU device to test (e.g. GPU.0, GPU.1). "
                             "Default: auto-detect and test all.")
    args = parser.parse_args()

    import openwakeword
    oww_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")

    import configparser
    cfg = configparser.ConfigParser()
    cfg.read(os.path.join(SCRIPT_DIR, "config", "jarvis.ini"))
    tts_voice_name = cfg.get("tts", "voice", fallback="en_US-ryan-high")

    # --- Detect GPU devices ---
    probe_path = os.path.join(oww_dir, "melspectrogram.onnx")
    if args.gpu:
        gpu_devices = [args.gpu]
        print(f"Testing specified GPU device: {args.gpu}")
    else:
        print("Detecting OpenVINO GPU devices...", end=" ", flush=True)
        gpu_devices = detect_gpu_devices(probe_path)
        if gpu_devices:
            print(f"found {len(gpu_devices)}: {', '.join(gpu_devices)}")
        else:
            print("none found")

    # --- Build provider configs ---
    providers = {"CPU": ["CPUExecutionProvider"]}
    if "OpenVINOExecutionProvider" in ort.get_available_providers():
        providers["OV-CPU"] = [
            ("OpenVINOExecutionProvider", {"device_type": "CPU"}),
            "CPUExecutionProvider",
        ]
    for dev in gpu_devices:
        providers[f"OV-{dev}"] = [
            ("OpenVINOExecutionProvider", {"device_type": dev}),
            "CPUExecutionProvider",
        ]

    # --- Build model definitions with realistic inputs ---
    models = []

    models.append({
        "name": "melspectrogram",
        "desc": "80ms audio chunk",
        "path": os.path.join(oww_dir, "melspectrogram.onnx"),
        "feed": lambda s: {"input": np.random.randn(1, 1280).astype(np.float32)},
    })

    models.append({
        "name": "embedding",
        "desc": "76x32 melspec window",
        "path": os.path.join(oww_dir, "embedding_model.onnx"),
        "feed": lambda s: {"input_1": np.random.randn(1, 76, 32, 1).astype(np.float32)},
    })

    models.append({
        "name": "hey_jarvis",
        "desc": "16x96 embedding window",
        "path": os.path.join(oww_dir, "hey_jarvis_v0.1.onnx"),
        "feed": lambda s: {"x.1": np.random.randn(1, 16, 96).astype(np.float32)},
    })

    models.append({
        "name": "silero_vad",
        "desc": "32ms audio chunk",
        "path": os.path.join(oww_dir, "silero_vad.onnx"),
        "feed": lambda s: {
            "input": np.random.randn(1, 512).astype(np.float32),
            "sr": np.array(16000, dtype=np.int64),
            "h": np.zeros((2, 1, 64), dtype=np.float32),
            "c": np.zeros((2, 1, 64), dtype=np.float32),
        },
    })

    tts_path = os.path.join(SCRIPT_DIR, "voices", f"{tts_voice_name}.onnx")
    if os.path.exists(tts_path):
        def _tts_feed(sess):
            feed = {}
            for inp in sess.get_inputs():
                shape = [s if isinstance(s, int) else 1 for s in inp.shape]
                if "int" in inp.type:
                    shape[-1] = 130
                    feed[inp.name] = np.ones(shape, dtype=np.int64)
                else:
                    feed[inp.name] = np.random.randn(*shape).astype(np.float32)
            return feed

        models.append({
            "name": "piper_tts",
            "desc": "130 phoneme IDs (~1 sentence)",
            "path": tts_path,
            "feed": _tts_feed,
        })

    # --- Run benchmarks ---
    prov_names = list(providers.keys())
    header = f"{'Model':<16} {'Input':<28}"
    units  = f"{'':16} {'':28}"
    for p in prov_names:
        header += f" {p:>12}"
        units  += f" {'(ms)':>12}"
    print()
    print(header)
    print(units)
    print("-" * len(header))

    for model in models:
        results = {}
        for prov_name, prov_list in providers.items():
            sess = try_load(model["path"], prov_list)
            if sess is None:
                results[prov_name] = "FAIL"
                continue
            try:
                feed = model["feed"](sess)
                ms = bench(sess, feed)
                results[prov_name] = f"{ms:.2f}"
            except Exception:
                results[prov_name] = "ERR"

        row = f"{model['name']:<16} {model['desc']:<28}"
        for p in prov_names:
            row += f" {results.get(p, 'N/A'):>12}"
        print(row)

    print()
    print(f"({BENCH_ITERS} iterations after {WARMUP_ITERS} warmup)")


if __name__ == "__main__":
    main()
