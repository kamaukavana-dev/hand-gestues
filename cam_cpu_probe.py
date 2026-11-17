#!/usr/bin/env python3
import argparse
import os
import time

import cv2
import psutil


def open_camera(mode):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap.release()
        return None
    if mode in ("D", "E"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap


def run_mode(cap, mode, seconds, process):
    process.cpu_percent(interval=None)
    start_times = process.cpu_times()
    started = time.monotonic()
    next_sample = started + 1.0
    timestamps = []
    retrieved_timestamps = []
    cpu_samples = []
    retrieve_count = 0

    while True:
        now = time.monotonic()
        if now - started >= seconds:
            break

        if mode in ("A", "D"):
            ok, _ = cap.read()
            if not ok:
                break
            timestamp = time.monotonic()
            timestamps.append(timestamp)
            retrieved_timestamps.append(timestamp)
        elif mode == "B":
            ok = cap.grab()
            if not ok:
                break
            timestamps.append(time.monotonic())
        else:
            ok = cap.grab()
            if not ok:
                break
            timestamps.append(time.monotonic())
            retrieve_count += 1
            retrieve_every = 2 if mode == "C" else 3
            if retrieve_count % retrieve_every == 0:
                ok, _ = cap.retrieve()
                if not ok:
                    break
                retrieved_timestamps.append(time.monotonic())

        now = time.monotonic()
        if now >= next_sample:
            cpu_percent = process.cpu_percent(interval=None)
            rss_mb = process.memory_info().rss / (1024 * 1024)
            cpu_samples.append((now - started, cpu_percent, rss_mb))
            next_sample += 1.0
            if next_sample <= now:
                next_sample = now + 1.0

    end_times = process.cpu_times()
    return (
        timestamps,
        retrieved_timestamps,
        cpu_samples,
        end_times.user - start_times.user,
        end_times.system - start_times.system,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["A", "B", "C", "D", "E", "F"],
        required=True,
    )
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument(
        "--threads",
        choices=["default", "1"],
        default="default",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0")
    if args.threads == "1":
        cv2.setNumThreads(1)

    print(f"OpenCV threads: {cv2.getNumThreads()}", flush=True)
    print(f"CPU count: {os.cpu_count()}", flush=True)

    process = psutil.Process()
    cap = None
    try:
        cap = open_camera(args.mode)
        if cap is None:
            print("CAMERA OPEN FAILED")
            return 1
        print("CAMERA LIVE")
        (
            timestamps,
            retrieved_timestamps,
            cpu_samples,
            user_seconds,
            system_seconds,
        ) = run_mode(
            cap,
            args.mode,
            args.seconds,
            process,
        )
    finally:
        if cap is not None:
            cap.release()
            print("CAMERA RELEASED")

    if len(timestamps) >= 2:
        fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
    else:
        fps = 0.0

    if len(retrieved_timestamps) >= 2:
        retrieved_fps = (len(retrieved_timestamps) - 1) / (
            retrieved_timestamps[-1] - retrieved_timestamps[0]
        )
    else:
        retrieved_fps = 0.0

    valid = [sample for sample in cpu_samples if sample[0] > 2.0]
    if valid:
        cpu_mean = sum(sample[1] for sample in valid) / len(valid)
    else:
        cpu_mean = None

    rss_mb = (
        cpu_samples[-1][2]
        if cpu_samples
        else process.memory_info().rss / (1024 * 1024)
    )

    print(f"Mode: {args.mode}")
    print(f"Measured FPS: {fps:.2f}")
    if cpu_mean is None:
        print("CPU% mean (first 2s excluded): n/a")
    else:
        print(f"CPU% mean (first 2s excluded): {cpu_mean:.2f}")
    print(f"User CPU seconds: {user_seconds:.2f}")
    print(f"System CPU seconds: {system_seconds:.2f}")
    if cpu_mean is not None and fps > 0:
        cpu_ms = cpu_mean / 100.0 / fps * 1000.0
        print(f"CPU ms per captured frame: {cpu_ms:.2f}")
    else:
        print("CPU ms per captured frame: n/a")
    if cpu_mean is not None and retrieved_fps > 0:
        cpu_ms_retrieved = cpu_mean / 100.0 / retrieved_fps * 1000.0
        print(f"CPU ms per retrieved frame: {cpu_ms_retrieved:.2f}")
    else:
        print("CPU ms per retrieved frame: n/a")
    print(f"RSS MB: {rss_mb:.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
