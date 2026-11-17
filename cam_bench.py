#!/usr/bin/env python3
import argparse
import threading
import time

import cv2
import psutil

from capture import CameraStream


def thread_cpu_rows(process):
    names = {
        thread.native_id: thread.name
        for thread in threading.enumerate()
        if thread.native_id is not None
    }
    rows = []
    for thread in process.threads():
        name = names.get(thread.id, f"unknown-{thread.id}")
        total = thread.user_time + thread.system_time
        rows.append(
            (name, thread.id, thread.user_time, thread.system_time, total)
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0")

    process = psutil.Process()
    samples = []
    thread_rows = []
    window_fps_values = []
    stream = CameraStream(
        index=0,
        width=640,
        height=480,
        stall_timeout_s=3.0,
        max_reopens=3,
    )

    try:
        stream.start()
        started = time.monotonic()
        next_sample = started + 1.0
        next_window = started + 5.0
        window_start_frames = stream.stats()["frames_captured"]
        window_start_ts = started
        process.cpu_percent(interval=None)
        previous_times = process.cpu_times()

        while next_sample <= started + args.seconds + 1e-9:
            now = time.monotonic()
            sleep_for = next_sample - now
            if sleep_for > 0:
                time.sleep(sleep_for)

            now = time.monotonic()
            elapsed = now - started
            stats = stream.stats()
            latest = stream.get_latest()
            current_times = process.cpu_times()
            user_delta = current_times.user - previous_times.user
            system_delta = current_times.system - previous_times.system
            previous_times = current_times
            cpu_percent = process.cpu_percent(interval=None)
            rss_mb = process.memory_info().rss / (1024 * 1024)
            samples.append(
                {
                    "elapsed": elapsed,
                    "cpu": cpu_percent,
                    "user": user_delta,
                    "system": system_delta,
                    "rss": rss_mb,
                }
            )

            line = (
                f"t={elapsed:.1f}s CPU={cpu_percent:.1f}% "
                f"user={user_delta:.2f}s system={system_delta:.2f}s "
                f"RSS={rss_mb:.1f} MB"
            )

            if now >= next_window or elapsed >= args.seconds - 1e-9:
                window_fps = None
                if latest is not None:
                    window_seconds = latest[1] - window_start_ts
                    frames_delta = (
                        stats["frames_captured"] - window_start_frames
                    )
                    if frames_delta > 0 and window_seconds > 0:
                        window_fps = frames_delta / window_seconds
                if window_fps is None:
                    line += " window_fps=n/a"
                else:
                    line += f" window_fps={window_fps:.2f}"
                    window_fps_values.append(window_fps)
                if latest is not None:
                    try:
                        brightness = float(latest[0].mean())
                    except Exception:
                        brightness = None
                else:
                    brightness = None
                auto_exposure = stream.read_cap_prop(
                    cv2.CAP_PROP_AUTO_EXPOSURE
                )
                exposure = stream.read_cap_prop(cv2.CAP_PROP_EXPOSURE)
                brightness_text = (
                    "n/a" if brightness is None else f"{brightness:.2f}"
                )
                auto_exposure_text = (
                    "n/a"
                    if auto_exposure is None
                    else f"{auto_exposure:.2f}"
                )
                exposure_text = (
                    "n/a" if exposure is None else f"{exposure:.2f}"
                )
                line += (
                    f" brightness={brightness_text} "
                    f"auto_exposure={auto_exposure_text} "
                    f"exposure={exposure_text}"
                )
                window_start_frames = stats["frames_captured"]
                window_start_ts = (
                    latest[1]
                    if latest is not None and latest[1] > window_start_ts
                    else now
                )
                while next_window <= now:
                    next_window += 5.0

            print(line)
            if stats["fatal_error"] is not None:
                break
            next_sample += 1.0
    except KeyboardInterrupt:
        pass
    finally:
        thread_rows = thread_cpu_rows(process)
        stream.stop()

    stats = stream.stats()
    valid = [sample for sample in samples if sample["elapsed"] > 2.0]

    print(f"Label: {args.label}")
    print(f"Measured FPS: {stats['fps']:.2f}")
    if valid:
        cpu_values = [sample["cpu"] for sample in valid]
        rss_values = [sample["rss"] for sample in valid]
        print(f"CPU% mean: {sum(cpu_values) / len(cpu_values):.2f}")
        print(f"CPU% peak: {max(cpu_values):.2f}")
        print(f"RSS MB mean: {sum(rss_values) / len(rss_values):.2f}")
        print(f"RSS MB final: {rss_values[-1]:.2f}")
    else:
        print("CPU% mean: n/a")
        print("CPU% peak: n/a")
        print("RSS MB mean: n/a")
        print("RSS MB final: n/a")
    print(f"Stall count: {stats['stall_count']}")
    print(f"Reopen count: {stats['reopen_count']}")
    print(f"Fatal error: {stats['fatal_error']}")
    print("Per-thread CPU (sampled before release):")
    for name, tid, user, system, total in sorted(
        thread_rows,
        key=lambda row: row[4],
        reverse=True,
    ):
        print(
            f"  {name} (tid={tid}): user={user:.2f}s "
            f"system={system:.2f}s total={total:.2f}s"
        )
    if thread_rows:
        top = max(thread_rows, key=lambda row: row[4])
        print(f"Top CPU thread: {top[0]} (tid={top[1]}) total={top[4]:.2f}s")
    if window_fps_values:
        print(f"Window FPS min: {min(window_fps_values):.2f}")
        print(f"Window FPS max: {max(window_fps_values):.2f}")
    else:
        print("Window FPS min: n/a")
        print("Window FPS max: n/a")
    rss_at_5 = next(
        (
            sample["rss"]
            for sample in samples
            if sample["elapsed"] >= 5.0 - 1e-9
        ),
        None,
    )
    if rss_at_5 is not None and samples:
        rss_change = samples[-1]["rss"] - rss_at_5
        print(f"RSS change MB (final - t=5s): {rss_change:.2f}")
    else:
        print("RSS change MB (final - t=5s): n/a")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
