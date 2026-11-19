#!/usr/bin/env python3
"""Offline gesture-model load and synthetic-frame test.
No camera, no network, no files written.
API: mp.tasks.vision / mp.tasks.BaseOptions only.
"""
import statistics
import sys
import threading
import time

import cv2
import numpy as np
import psutil
import mediapipe as mp

MODEL_PATH = "models/gesture_recognizer.task"


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


def thread_count():
    return threading.active_count()


def main():
    # --- Pre-load info ---
    proc = psutil.Process()
    print("=" * 60)
    print("SECTION 1: Pre-load stats")
    print("=" * 60)
    print(f"mediapipe version : {mp.__version__}")
    print(f"cv2 version       : {cv2.__version__}")
    print(f"numpy version     : {np.__version__}")
    print(f"Thread count      : {thread_count()}")
    print(f"RSS MB (before)   : {rss_mb():.2f}")

    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    RunningMode = mp.tasks.vision.RunningMode
    BaseOptions = mp.tasks.BaseOptions

    # --- IMAGE mode ---
    print()
    print("=" * 60)
    print("SECTION 2: IMAGE mode – construct, recognize, close")
    print("=" * 60)

    options_image = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
    )

    t0 = time.monotonic()
    recognizer_image = GestureRecognizer.create_from_options(options_image)
    construct_ms = (time.monotonic() - t0) * 1000
    print(f"Construction time : {construct_ms:.1f} ms")
    print(f"RSS MB (after construct) : {rss_mb():.2f}")

    try:
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=black_frame)
        result = recognizer_image.recognize(mp_image)
        gestures = result.gestures
        print(f"Gestures from black frame: {gestures}")
    finally:
        recognizer_image.close()
        print("IMAGE recognizer closed.")

    # --- VIDEO mode ---
    print()
    print("=" * 60)
    print("SECTION 3: VIDEO mode – 150 synthetic frames @ 60ms pace")
    print("=" * 60)

    options_video = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )

    recognizer_video = GestureRecognizer.create_from_options(options_video)

    per_call_ms = []
    rng = np.random.default_rng(42)

    # CPU sampling thread
    cpu_samples = []
    sampling_active = True

    def cpu_sampler():
        time.sleep(2.0)  # exclude first 2 s
        while sampling_active:
            cpu_samples.append(proc.cpu_percent(interval=1.0))

    sampler_thread = threading.Thread(target=cpu_sampler, daemon=True)

    try:
        last_ts = int(time.monotonic() * 1000)
        deadline = time.monotonic()
        proc_times_start = proc.cpu_times()
        loop_start = time.monotonic()

        sampler_thread.start()

        for i in range(150):
            deadline += 0.060  # 60 ms per frame → ~16.65 FPS

            # Alternating black / random-noise frame
            if i % 2 == 0:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            ts = int(time.monotonic() * 1000)
            if ts <= last_ts:
                ts = last_ts + 1
            last_ts = ts

            t_call = time.monotonic()
            recognizer_video.recognize_for_video(mp_image, ts)
            per_call_ms.append((time.monotonic() - t_call) * 1000)

            # Pace to deadline
            now = time.monotonic()
            if now < deadline:
                time.sleep(deadline - now)

        loop_elapsed = time.monotonic() - loop_start
        proc_times_end = proc.cpu_times()

    finally:
        sampling_active = False
        sampler_thread.join(timeout=3.0)
        recognizer_video.close()
        print("VIDEO recognizer closed.")

    # --- Results ---
    print()
    print("=" * 60)
    print("SECTION 4: Per-call latency (ms)")
    print("=" * 60)
    sorted_ms = sorted(per_call_ms)
    n = len(sorted_ms)
    p95_idx = int(n * 0.95)
    print(f"Frames         : {n}")
    print(f"Loop wall time : {loop_elapsed:.2f} s")
    print(f"Mean ms        : {statistics.mean(per_call_ms):.3f}")
    print(f"Median ms      : {statistics.median(per_call_ms):.3f}")
    print(f"p95 ms         : {sorted_ms[p95_idx]:.3f}")
    print(f"Max ms         : {max(per_call_ms):.3f}")

    print()
    print("=" * 60)
    print("SECTION 5: CPU / memory")
    print("=" * 60)
    user_delta = proc_times_end.user - proc_times_start.user
    sys_delta = proc_times_end.system - proc_times_start.system
    print(f"CPU% samples (excl. first 2s) : {cpu_samples}")
    if cpu_samples:
        print(f"CPU% mean                     : {statistics.mean(cpu_samples):.2f}")
    print(f"User CPU seconds              : {user_delta:.3f}")
    print(f"System CPU seconds            : {sys_delta:.3f}")
    print(f"RSS MB (end)                  : {rss_mb():.2f}")
    print(f"Thread count (end)            : {thread_count()}")
    print()
    print("DONE.")


if __name__ == "__main__":
    main()
