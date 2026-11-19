#!/usr/bin/env python3
"""Live gesture recognizer with cost telemetry and rate limiting.
Prints classifications on state/label change; logs 5-second performance windows;
outputs full-run summary upon termination.

NO camera frame writes, NO OS actions, NO input injection, NO network calls.
"""
import argparse
import collections
import os
import statistics
import sys
import time

import cv2
import mediapipe as mp
import psutil

from capture import CameraStream
from gesture_logic import RateLimiter, WindowStats, classify

MODEL_PATH = "models/gesture_recognizer.task"
ALLOWED_LOG_DIR = os.path.realpath(os.path.expanduser("~/hand-gestures/logs"))


def validate_log_csv(csv_path_str):
    if not csv_path_str:
        return None
    abs_path = os.path.realpath(os.path.expanduser(csv_path_str))
    # Refuse paths outside ~/hand-gestures/logs/
    try:
        common = os.path.commonpath([abs_path, ALLOWED_LOG_DIR])
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Path must be located inside {ALLOWED_LOG_DIR}: {csv_path_str}"
        )
    if common != ALLOWED_LOG_DIR or abs_path == ALLOWED_LOG_DIR:
        raise argparse.ArgumentTypeError(
            f"Path must be located inside {ALLOWED_LOG_DIR}: {csv_path_str}"
        )
    return abs_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run live gesture recognition on CameraStream with cost telemetry."
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=60.0,
        help="Run duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=8.0,
        help="Maximum recognition inferences per second (default: 8)",
    )
    parser.add_argument(
        "--log-csv",
        type=validate_log_csv,
        default=None,
        help="Optional path to output CSV log file (must be inside ~/hand-gestures/logs/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isfile(MODEL_PATH):
        sys.stderr.write(f"Error: model file not found at {MODEL_PATH}\n")
        sys.exit(1)

    csv_file = None
    if args.log_csv:
        os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
        csv_file = open(args.log_csv, "w", buffering=1)
        csv_file.write("ts_ms,state,label,score,handedness,infer_ms\n")

    # Configure MediaPipe Tasks Vision GestureRecognizer
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    RunningMode = mp.tasks.vision.RunningMode
    BaseOptions = mp.tasks.BaseOptions

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
    )

    proc = psutil.Process()
    # Prime cpu_percent baseline
    proc.cpu_percent(interval=None)

    recognizer = GestureRecognizer.create_from_options(options)

    # Telemetry and state trackers
    limiter = RateLimiter(max_fps=args.max_fps)
    window_stats = WindowStats()
    total_stats = WindowStats()
    per_label_counts = collections.Counter()

    last_processed_frame_ts = None
    last_mp_ts_ms = -1
    last_state_label = None

    limiter_skips_window = 0
    limiter_skips_total = 0

    start_time = time.monotonic()
    window_start = start_time

    print(f"Starting recognize_live: seconds={args.seconds:g}, max_fps={args.max_fps:g}")
    if args.log_csv:
        print(f"Logging CSV to: {args.log_csv}")

    try:
        with CameraStream(cv_threads=1) as stream:
            while True:
                now = time.monotonic()
                if now - start_time >= args.seconds:
                    break

                if stream.fatal_error:
                    sys.stderr.write(f"Camera fatal error: {stream.fatal_error}\n")
                    break

                latest = stream.get_latest()
                if latest is None:
                    time.sleep(0.005)
                    continue

                frame, frame_ts = latest
                if last_processed_frame_ts is not None and frame_ts == last_processed_frame_ts:
                    time.sleep(0.005)
                    continue

                # Rate limiter gate
                if not limiter.allow():
                    limiter_skips_window += 1
                    limiter_skips_total += 1
                    last_processed_frame_ts = frame_ts
                    continue

                last_processed_frame_ts = frame_ts

                # Frame format conversion (BGR -> RGB -> mp.Image)
                t_conv_0 = time.perf_counter()
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                convert_ms = (time.perf_counter() - t_conv_0) * 1000.0

                # Monotonic timestamp generation in milliseconds
                ts_ms = int(frame_ts * 1000)
                if ts_ms <= last_mp_ts_ms:
                    ts_ms = last_mp_ts_ms + 1
                last_mp_ts_ms = ts_ms

                # Inference timing
                t_inf_0 = time.perf_counter()
                result = recognizer.recognize_for_video(mp_image, ts_ms)
                infer_ms = (time.perf_counter() - t_inf_0) * 1000.0

                # Classification & change detection
                state, label, score, handedness = classify(result)

                if (state, label) != last_state_label:
                    last_state_label = (state, label)
                    elapsed_s = time.monotonic() - start_time
                    if state == "GESTURE":
                        print(f"t={elapsed_s:.1f}s GESTURE {label} {score:.2f} {handedness}")
                    elif state == "NONE":
                        print(f"t={elapsed_s:.1f}s NONE")
                    else:
                        print(f"t={elapsed_s:.1f}s NO_HAND")

                # Accumulate telemetry
                window_stats.add(infer_ms, convert_ms, state, label)
                total_stats.add(infer_ms, convert_ms, state, label)
                if label:
                    per_label_counts[label] += 1
                else:
                    per_label_counts[state] += 1

                # Optional CSV logging (no image frames)
                if csv_file is not None:
                    csv_file.write(
                        f"{ts_ms},{state},{label},{score:.4f},{handedness},{infer_ms:.2f}\n"
                    )

                # Periodic 5-second status reporting
                now_mono = time.monotonic()
                if now_mono - window_start >= 5.0:
                    w_duration = now_mono - window_start
                    w_count = window_stats.count()
                    w_fps = w_count / w_duration if w_duration > 0 else 0.0
                    inf_mean = window_stats.mean(window_stats.infer_times)
                    inf_p95 = window_stats.p95(window_stats.infer_times)
                    conv_mean = window_stats.mean(window_stats.convert_times)
                    cpu_pct = proc.cpu_percent(interval=None)
                    rss_mb = proc.memory_info().rss / (1024 * 1024)
                    cam_stats = stream.stats()
                    cam_fps = cam_stats.get("fps", 0.0)

                    state_fracs = window_stats.state_fractions()
                    state_str = ", ".join(
                        f"{k}:{v * 100:.1f}%" for k, v in sorted(state_fracs.items())
                    )
                    if not state_str:
                        state_str = "no_samples"

                    print(
                        f"[STATUS t={now_mono - start_time:.1f}s] "
                        f"DetFPS={w_fps:.2f} | "
                        f"Infer(mean={inf_mean:.1f}ms, p95={inf_p95:.1f}ms) | "
                        f"Conv={conv_mean:.2f}ms | "
                        f"CPU={cpu_pct:.1f}% | "
                        f"RSS={rss_mb:.1f}MB | "
                        f"CamFPS={cam_fps:.2f} | "
                        f"States=[{state_str}] | "
                        f"LimiterSkips={limiter_skips_window}"
                    )
                    window_stats.reset()
                    limiter_skips_window = 0
                    window_start = now_mono

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Guarantee resource release on all paths
        try:
            recognizer.close()
        except Exception:
            pass

        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

        # Full-run summary
        total_duration = time.monotonic() - start_time
        total_frames = total_stats.count()
        det_fps = total_frames / total_duration if total_duration > 0 else 0.0

        print("\n" + "=" * 60)
        print("FULL RUN SUMMARY")
        print("=" * 60)
        print(f"Total time            : {total_duration:.2f} s")
        print(f"Frames detected/proc  : {total_frames}")
        print(f"Achieved detection FPS: {det_fps:.2f}")
        print(f"Limiter skipped frames: {limiter_skips_total}")
        if total_frames > 0:
            print(
                f"Inference latency ms  : mean={total_stats.mean(total_stats.infer_times):.2f}, "
                f"median={statistics.median(total_stats.infer_times):.2f}, "
                f"p95={total_stats.p95(total_stats.infer_times):.2f}, "
                f"max={max(total_stats.infer_times):.2f}"
            )
            print(
                f"Convert latency ms    : mean={total_stats.mean(total_stats.convert_times):.2f}, "
                f"max={max(total_stats.convert_times):.2f}"
            )
        else:
            print("Inference latency ms  : no frames processed")
        print(f"Final Process RSS     : {proc.memory_info().rss / (1024 * 1024):.2f} MB")
        print("Per-label counts:")
        for lbl, count in sorted(per_label_counts.items(), key=lambda x: -x[1]):
            print(f"  {lbl}: {count}")
        print("=" * 60)


if __name__ == "__main__":
    main()
