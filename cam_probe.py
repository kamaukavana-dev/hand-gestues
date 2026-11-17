#!/usr/bin/env python3
import argparse
import sys
import time

import cv2

CONFIGS = [
    ("default format, no FPS request", None, None),
    ("default format, 30 FPS request", None, 30),
    ("MJPG, no FPS request", "MJPG", None),
    ("MJPG, 30 FPS request", "MJPG", 30),
]


def fourcc_to_string(value):
    chars = "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))
    if all(32 <= ord(char) <= 126 for char in chars):
        return chars
    return "n/a"


def open_configured(fourcc, fps_request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None
    if fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if fps_request is not None:
        cap.set(cv2.CAP_PROP_FPS, fps_request)
    return cap


def probe_config(label, fourcc, fps_request):
    cap = None
    try:
        cap = open_configured(fourcc, fps_request)
        if cap is None:
            print(f"Config: {label}")
            print("  OPEN FAILED")
            print()
            return None

        reported_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        reported_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Config: {label}")
        print(f"  FOURCC: {fourcc_to_string(reported_fourcc)} ({reported_fourcc})")
        print(f"  CAP_PROP_FPS: {reported_fps:.3f}")
        print(f"  Resolution: {width}x{height}")

        for index in range(10):
            ok, _ = cap.read()
            if not ok:
                print(f"  ERROR: warm-up read {index + 1} failed")
                print()
                return None

        timestamps = []
        for _ in range(150):
            ok, _ = cap.read()
            if not ok:
                print("  ERROR: capture read failed")
                break
            timestamps.append(time.monotonic())

        if len(timestamps) < 2:
            print(f"  Captured: {len(timestamps)} frames")
            print("  Measured average FPS: n/a")
            print("  Inter-frame gap: n/a")
            print()
            return None

        gaps_ms = [
            (timestamps[index] - timestamps[index - 1]) * 1000.0
            for index in range(1, len(timestamps))
        ]
        average_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])

        print(f"  Captured: {len(timestamps)} frames")
        print(f"  Measured average FPS: {average_fps:.2f}")
        print(
            f"  Inter-frame gap min/max: "
            f"{min(gaps_ms):.2f} / {max(gaps_ms):.2f} ms"
        )
        print()
        return {
            "label": label,
            "fourcc": fourcc,
            "fps_request": fps_request,
            "average_fps": average_fps,
        }
    finally:
        if cap is not None:
            cap.release()
            print(f"  Camera released: {label}")
            print()


def run_cpu(best):
    try:
        import psutil
    except ImportError:
        print("ERROR: psutil is not installed; run without --cpu or install psutil")
        return 1

    cap = None
    try:
        cap = open_configured(best["fourcc"], best["fps_request"])
        if cap is None:
            print("CPU: OPEN FAILED")
            print()
            return 1

        print(f"CPU measurement config: {best['label']}")
        for index in range(10):
            ok, _ = cap.read()
            if not ok:
                print(f"ERROR: warm-up read {index + 1} failed")
                print()
                return 1

        process = psutil.Process()
        process.cpu_percent(interval=None)
        started = time.monotonic()
        next_sample = started + 1.0
        samples = []
        frame_count = 0

        while True:
            now = time.monotonic()
            if now - started >= 10.0:
                break

            ok, _ = cap.read()
            if not ok:
                print("ERROR: capture read failed")
                print()
                return 1
            frame_count += 1

            now = time.monotonic()
            if now >= next_sample:
                cpu_percent = process.cpu_percent(interval=None)
                rss_mb = process.memory_info().rss / (1024 * 1024)
                samples.append((now - started, cpu_percent, rss_mb))
                print(
                    f"  t={now - started:.1f}s "
                    f"CPU={cpu_percent:.1f}% RSS={rss_mb:.1f} MB"
                )
                next_sample += 1.0

        valid = [sample for sample in samples if sample[0] > 2.0]
        print(f"  Frames captured: {frame_count}")
        if valid:
            average_cpu = sum(sample[1] for sample in valid) / len(valid)
            average_rss = sum(sample[2] for sample in valid) / len(valid)
            print(f"  Process CPU% (avg, first 2s excluded): {average_cpu:.2f}")
            print(f"  RSS MB (avg, first 2s excluded): {average_rss:.2f}")
            print(f"  RSS MB (final): {valid[-1][2]:.2f}")
        else:
            print("  Process CPU%: n/a")
            print("  RSS MB: n/a")
        print()
        return 0
    finally:
        if cap is not None:
            cap.release()
            print("  Camera released")
            print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    results = []
    for label, fourcc, fps_request in CONFIGS:
        result = probe_config(label, fourcc, fps_request)
        if result is not None:
            results.append(result)

    if not results:
        print("ERROR: no camera configuration succeeded")
        return 1

    best = max(results, key=lambda item: item["average_fps"])
    print(f"Best config: {best['label']} ({best['average_fps']:.2f} FPS)")
    print()

    if args.cpu:
        return run_cpu(best)
    return 0


if __name__ == "__main__":
    sys.exit(main())
