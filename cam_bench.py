#!/usr/bin/env python3
import argparse
import time

import psutil

from capture import CameraStream


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than 0")

    process = psutil.Process()
    samples = []
    stream = CameraStream(
        index=0,
        width=640,
        height=480,
        stall_timeout_s=3.0,
        max_reopens=3,
    )

    try:
        stream.start()
        process.cpu_percent(interval=None)
        started = time.monotonic()
        next_sample = started + 1.0

        while True:
            now = time.monotonic()
            elapsed = now - started
            if elapsed >= args.seconds:
                break
            if stream.stats()["fatal_error"] is not None:
                break

            sleep_for = min(next_sample - now, args.seconds - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

            now = time.monotonic()
            elapsed = now - started
            cpu_percent = process.cpu_percent(interval=None)
            rss_mb = process.memory_info().rss / (1024 * 1024)
            samples.append((elapsed, cpu_percent, rss_mb))
            print(
                f"t={elapsed:.1f}s CPU={cpu_percent:.1f}% RSS={rss_mb:.1f} MB"
            )
            next_sample += 1.0
            if next_sample <= now:
                next_sample = now + 1.0
    finally:
        stream.stop()

    stats = stream.stats()
    valid = [sample for sample in samples if sample[0] > 2.0]

    print(f"Label: {args.label}")
    print(f"Measured FPS: {stats['fps']:.2f}")
    if valid:
        cpu_values = [sample[1] for sample in valid]
        rss_values = [sample[2] for sample in valid]
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
