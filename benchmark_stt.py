#!/usr/bin/env python3
"""Benchmark STT inference: faster-whisper (CPU) vs accelerated backends.

Uses the same benchmark functions as jarvis.py's auto-detect to ensure
consistent results. Run this to manually compare devices.

Usage:
    python benchmark_stt.py              # Benchmark all available backends/devices
    python benchmark_stt.py --gpu GPU.1  # Benchmark only a specific GPU
    python benchmark_stt.py --duration 8 # Use 8s test audio
"""

import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Suppress noisy warnings
import warnings
warnings.filterwarnings("ignore", message=".*CUDAExecutionProvider.*")


def main():
    parser = argparse.ArgumentParser(description="Benchmark STT backends")
    parser.add_argument("--gpu", type=str, default=None,
                        help="Specific GPU device to test (e.g. GPU.0, GPU.1)")
    parser.add_argument("--model", type=str, default=None,
                        help="Whisper model name (default: from jarvis.ini)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Test audio duration in seconds (default: 5.0)")
    args = parser.parse_args()

    from jarvis import (
        _HAS_OPENVINO,
        _detect_gpu_devices,
        _generate_bench_audio,
        _bench_faster_whisper,
        STT_MODEL,
    )

    model_name = args.model or STT_MODEL
    audio_path = _generate_bench_audio(args.duration)

    print(f"Model: whisper-{model_name}")
    print(f"Test audio: {args.duration}s sine wave")
    print()

    results = {}

    # faster-whisper CPU (always available)
    print("[faster-whisper CPU]", flush=True)
    fw_time = _bench_faster_whisper(model_name, audio_path)
    if fw_time is not None:
        results["faster-whisper CPU"] = fw_time * 1000
        print(f"  {fw_time*1000:.0f} ms")
    print()

    # Detect GPU devices
    all_gpus = _detect_gpu_devices()
    if all_gpus:
        print(f"Detected GPUs: {', '.join(f'{b}:{d}' for b, d in all_gpus)}")
    else:
        print("No GPU devices detected")
    print()

    # OpenVINO benchmarks
    if _HAS_OPENVINO:
        from jarvis import _ensure_openvino_cache, _bench_openvino_device

        if args.gpu:
            ov_devices = [args.gpu]
        else:
            ov_devices = [d for b, d in all_gpus if b == "openvino"]

        try:
            cache_dir = _ensure_openvino_cache(model_name)
            for dev in ["CPU"] + ov_devices:
                label = f"OpenVINO {dev}"
                print(f"[{label}]", flush=True)
                ov_time = _bench_openvino_device(cache_dir, dev, audio_path)
                if ov_time is not None:
                    results[label] = ov_time * 1000
                    print(f"  {ov_time*1000:.0f} ms")
                print()
        except Exception as e:
            print(f"  OpenVINO setup failed: {e}\n")
    else:
        print("OpenVINO not installed, skipping OpenVINO benchmarks\n")

    # CUDA benchmarks (future)
    # if _HAS_CUDA:
    #     ...

    os.unlink(audio_path)

    # Summary
    if results:
        print("=" * 55)
        print(f"{'Provider':<25} {'Mean (ms)':>10} {'vs best':>10}")
        print("-" * 55)
        best = min(results.values())
        for name, ms in sorted(results.items(), key=lambda x: x[1]):
            ratio = f"{ms/best:.2f}x"
            marker = " <--" if ms == best else ""
            print(f"{name:<25} {ms:>10.0f} {ratio:>10}{marker}")


if __name__ == "__main__":
    main()
