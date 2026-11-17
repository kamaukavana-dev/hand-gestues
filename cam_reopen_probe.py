#!/usr/bin/env python3
import argparse
import time

import cv2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, required=True)
    parser.add_argument("--delay", type=float, required=True)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    if args.cycles < 1:
        parser.error("--cycles must be greater than 0")
    if args.delay < 0:
        parser.error("--delay must be at least 0")
    if args.frames < 1:
        parser.error("--frames must be greater than 0")

    max_open_time = 0.0
    max_first_read_time = 0.0
    cap = None
    cycle = 0

    try:
        for cycle in range(1, args.cycles + 1):
            t_before_open = time.monotonic()
            print(
                f"cycle={cycle} before_open t={t_before_open:.6f}",
                flush=True,
            )
            cap = cv2.VideoCapture(0)
            t_after_open = time.monotonic()
            open_time = t_after_open - t_before_open
            max_open_time = max(max_open_time, open_time)
            print(
                f"cycle={cycle} after_open t={t_after_open:.6f} "
                f"open_s={open_time:.6f} opened={int(cap.isOpened())}",
                flush=True,
            )

            t_before_first_read = time.monotonic()
            print(
                f"cycle={cycle} before_first_read "
                f"t={t_before_first_read:.6f}",
                flush=True,
            )
            ok, _ = cap.read()
            t_after_first_read = time.monotonic()
            first_read_time = t_after_first_read - t_before_first_read
            max_first_read_time = max(max_first_read_time, first_read_time)
            print(
                f"cycle={cycle} after_first_read "
                f"t={t_after_first_read:.6f} "
                f"first_read_s={first_read_time:.6f} ok={int(bool(ok))}",
                flush=True,
            )

            frames_read = 1 if ok else 0
            if ok:
                for _ in range(1, args.frames):
                    ok, _ = cap.read()
                    if not ok:
                        break
                    frames_read += 1

            t_before_release = time.monotonic()
            print(
                f"cycle={cycle} before_release t={t_before_release:.6f}",
                flush=True,
            )
            cap.release()
            cap = None
            t_after_release = time.monotonic()
            print(
                f"cycle={cycle} after_release t={t_after_release:.6f} "
                f"frames_read={frames_read}",
                flush=True,
            )

            if cycle < args.cycles and args.delay > 0:
                time.sleep(args.delay)
    finally:
        if cap is not None:
            t_before_release = time.monotonic()
            print(
                f"cycle={cycle} before_release(finally) "
                f"t={t_before_release:.6f}",
                flush=True,
            )
            cap.release()
            t_after_release = time.monotonic()
            print(
                f"cycle={cycle} after_release(finally) "
                f"t={t_after_release:.6f}",
                flush=True,
            )

    print(f"Max open time: {max_open_time:.6f} s", flush=True)
    print(f"Max first-read time: {max_first_read_time:.6f} s", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
