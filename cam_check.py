#!/usr/bin/env python3
import argparse
import sys
import time
from collections import deque

import cv2


def open_camera():
    indices = [0, 1, 2, 3]
    for index in indices:
        if sys.platform == "win32" and index == 0:
            backends = [
                (cv2.CAP_DSHOW, "CAP_DSHOW"),
                (cv2.CAP_ANY, "default backend"),
            ]
        else:
            backends = [(cv2.CAP_ANY, "default backend")]

        for backend, backend_name in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                return cap, index, backend_name
            cap.release()

    return None, None, None


def fps_from_timestamps(timestamps):
    if len(timestamps) < 2:
        return 0.0
    elapsed = timestamps[-1] - timestamps[0]
    if elapsed <= 0:
        return 0.0
    return (len(timestamps) - 1) / elapsed


def run_live(cap, timeout):
    window_name = "cam_check"
    timestamps = deque()
    started = time.monotonic()

    while True:
        if time.monotonic() - started >= timeout:
            print(f"Timeout reached after {timeout:g} seconds")
            return True

        ok, frame = cap.read()
        if not ok:
            print("ERROR: Failed to read a frame")
            return False

        now = time.monotonic()
        timestamps.append(now)
        while len(timestamps) > 2 and now - timestamps[0] > 1.0:
            timestamps.popleft()

        fps = fps_from_timestamps(timestamps)
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window_name, frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            return True


def run_headless(cap, frames, timeout):
    started = time.monotonic()
    first_timestamp = None
    last_timestamp = None
    captured = 0

    for _ in range(frames):
        if time.monotonic() - started >= timeout:
            print(f"Timeout reached after {timeout:g} seconds")
            break

        ok, _ = cap.read()
        if not ok:
            print("ERROR: Failed to read a frame")
            break

        now = time.monotonic()
        if first_timestamp is None:
            first_timestamp = now
        last_timestamp = now
        captured += 1

    if captured >= 2:
        elapsed = last_timestamp - first_timestamp
        fps = (captured - 1) / elapsed if elapsed > 0 else 0.0
        print(f"Captured {captured} frames")
        print(f"Measured average FPS: {fps:.2f}")
    else:
        print(f"Captured {captured} frames")
        print("Measured average FPS: 0.00")

    return captured


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")
    if args.headless and (args.frames is None or args.frames < 1):
        parser.error("--headless requires --frames N with N >= 1")
    if not args.headless and args.frames is not None:
        parser.error("--frames can only be used with --headless")

    cap = None
    exit_code = 0

    try:
        cap, index, backend_name = open_camera()
        if cap is None:
            print("ERROR: Could not open camera index 0, 1, 2, or 3")
            return 1

        print("CAMERA LIVE")
        print(f"Camera index: {index}")
        print(f"Backend: {backend_name}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print("Requested resolution: 640x480")
        print(f"Camera reports: {width}x{height}")

        if args.headless:
            if run_headless(cap, args.frames, args.timeout) == 0:
                exit_code = 1
        else:
            if not run_live(cap, args.timeout):
                exit_code = 1
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if cap is not None:
            cap.release()
            print("CAMERA RELEASED")
        if not args.headless:
            cv2.destroyAllWindows()

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
