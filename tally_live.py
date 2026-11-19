#!/usr/bin/env python3
"""Ground-truth tally live tool with pacing, phase prompts, and telemetry.
Collects trials per target across GET_READY, SETTLE, HOLD, and RELAX phases.

NO camera frame writes, NO OS actions, NO input injection, NO network calls.
"""
import argparse
import collections
import os
import sys
import time

import cv2
import mediapipe as mp
import psutil

from capture import CameraStream
from gesture_logic import MonotonicMs, RateLimiter, classify
from tally_logic import (
    DEFAULT_TARGETS,
    PHASE_DURATIONS,
    PHASE_GET_READY,
    PHASE_HOLD,
    PHASE_RELAX,
    PHASE_SETTLE,
    build_schedule,
    debounce_sim_multi,
    summarize,
)

MODEL_PATH = "models/gesture_recognizer.task"
ALLOWED_LOG_DIR = os.path.realpath(os.path.expanduser("~/hand-gestures/logs"))


def validate_log_csv(csv_path_str):
    if not csv_path_str:
        return None
    abs_path = os.path.realpath(os.path.expanduser(csv_path_str))
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
        description="Run live ground-truth gesture collection with schedule and telemetry."
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=8,
        help="Number of trials per target (default: 8)",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=8.0,
        help="Maximum recognition inferences per second (default: 8)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for deterministic trial shuffle (optional)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default=None,
        help="Comma-separated target list (default: standard 6 targets)",
    )
    parser.add_argument(
        "--out",
        type=validate_log_csv,
        default=None,
        help="Optional path to output CSV log file (must be inside ~/hand-gestures/logs/)",
    )
    parser.add_argument(
        "--landmarks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record normalized landmark coordinates x0,y0,z0..x20,y20,z20 (default: on)",
    )
    return parser.parse_args()


def extract_landmarks(result):
    """Extracts 21 normalized landmarks as a comma-separated string:

    x0,y0,z0..x20,y20,z20. Empty fields if no hand detected.
    """
    hand_landmarks = getattr(result, "hand_landmarks", None)
    if not hand_landmarks or len(hand_landmarks) == 0 or len(hand_landmarks[0]) < 21:
        return ",".join([""] * 63)

    coords = []
    for lm in hand_landmarks[0][:21]:
        coords.extend([f"{lm.x:.5f}", f"{lm.y:.5f}", f"{lm.z:.5f}"])
    return ",".join(coords)


def print_summary_tables(recorded_rows, phase_infer_times, total_duration, total_frames, proc):
    """Prints summarize() and debounce_sim() tables plus cost telemetry."""
    print("\n" + "=" * 70)
    print("GROUND TRUTH TALLY SUMMARY")
    print("=" * 70)

    if not recorded_rows:
        print("No frames recorded.")
        return

    summary = summarize(recorded_rows)
    targets = summary["targets"]

    print("\n--- TARGET RECALL & PASS RATES ---")
    headers = ["Target", "Trials", "Rec@0.5", "Pass@0.5", "Rec@0.7", "Pass@0.7", "Rec@0.9", "Pass@0.9"]
    row_fmt = "{:<14} | {:>6} | {:>7} | {:>8} | {:>7} | {:>8} | {:>7} | {:>8}"
    print(row_fmt.format(*headers))
    print("-" * 75)

    for tgt, data in targets.items():
        tr = data["trials"]
        thrs = data["thresholds"]
        r5 = f"{thrs[0.5]['mean_recall']*100:.1f}%" if 0.5 in thrs else "-"
        p5 = f"{thrs[0.5]['pass_rate']*100:.1f}%" if 0.5 in thrs else "-"
        r7 = f"{thrs[0.7]['mean_recall']*100:.1f}%" if 0.7 in thrs else "-"
        p7 = f"{thrs[0.7]['pass_rate']*100:.1f}%" if 0.7 in thrs else "-"
        r9 = f"{thrs[0.9]['mean_recall']*100:.1f}%" if 0.9 in thrs else "-"
        p9 = f"{thrs[0.9]['pass_rate']*100:.1f}%" if 0.9 in thrs else "-"
        print(row_fmt.format(tgt, tr, r5, p5, r7, p7, r9, p9))

    print("\n--- REST EMISSIONS ---")
    em_parts = [f"thr={thr}: {cnt} frames" for thr, cnt in sorted(summary["rest_emissions"].items())]
    print(", ".join(em_parts))

    print("\n--- CONFUSION MATRIX (HOLD PHASE) ---")
    labels = summary["all_predicted_labels"]
    cm = summary["confusion_matrix"]
    cm_header = "{:<14} | " + " | ".join(f"{lb:>10}" for lb in labels)
    print(cm_header.format("Target \\ Pred"))
    print("-" * (17 + 13 * len(labels)))
    for tgt in targets.keys():
        counts = [f"{cm[tgt].get(lb, 0):>10}" for lb in labels]
        print(f"{tgt:<14} | " + " | ".join(counts))

    print("\n--- DEBOUNCE SIMULATION (thr=0.6) ---")
    deb_results = debounce_sim_multi(recorded_rows, thr=0.6)
    deb_header = "{:<8} | {:<12} | {:<12} | {:<12}"
    print(deb_header.format("Hold (ms)", "True Pos (%)", "Confusion", "REST Fires"))
    print("-" * 52)
    for res in deb_results:
        tp_str = f"{res['target_fired']}/{res['non_rest_trials']} ({res['target_fired_pct']*100:.1f}%)"
        print(
            deb_header.format(
                f"{res['hold_ms']}ms",
                tp_str,
                str(res["wrong_gesture_fired"]),
                str(res["rest_fires"]),
            )
        )

    print("\n--- COST TELEMETRY ---")
    det_fps = total_frames / total_duration if total_duration > 0 else 0.0
    print(f"Total time            : {total_duration:.2f} s")
    print(f"Total processed frames: {total_frames}")
    print(f"Achieved detection FPS: {det_fps:.2f}")
    print("Mean infer_ms per phase:")
    for ph in [PHASE_GET_READY, PHASE_SETTLE, PHASE_HOLD, PHASE_RELAX]:
        times = phase_infer_times.get(ph, [])
        m_time = (sum(times) / len(times)) if times else 0.0
        print(f"  {ph:<10}: {m_time:.2f} ms ({len(times)} frames)")

    cpu_pct = proc.cpu_percent(interval=None)
    rss_mb = proc.memory_info().rss / (1024 * 1024)
    print(f"Process CPU%          : {cpu_pct:.1f}%")
    print(f"Process RSS           : {rss_mb:.2f} MB")
    print("=" * 70)


def main():
    args = parse_args()

    if not os.path.isfile(MODEL_PATH):
        sys.stderr.write(f"Error: model file not found at {MODEL_PATH}\n")
        sys.exit(1)

    targets = [t.strip() for t in args.targets.split(",")] if args.targets else list(DEFAULT_TARGETS)
    schedule = build_schedule(targets=targets, trials=args.trials, seed=args.seed)
    total_trials = len(schedule)

    csv_file = None
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        csv_file = open(args.out, "w", buffering=1)
        base_cols = "ts_ms,trial,target,phase,state,label,score,handedness,infer_ms"
        if args.landmarks:
            lm_cols = [f"{coord}{i}" for i in range(21) for coord in ("x", "y", "z")]
            csv_file.write(f"{base_cols},{','.join(lm_cols)}\n")
        else:
            csv_file.write(f"{base_cols}\n")

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
    proc.cpu_percent(interval=None)

    recognizer = GestureRecognizer.create_from_options(options)

    limiter = RateLimiter(max_fps=args.max_fps)
    mono_ms = MonotonicMs()

    recorded_rows = []
    phase_infer_times = collections.defaultdict(list)
    total_frames = 0
    start_monotonic = time.monotonic()

    print(f"Starting tally_live: {total_trials} trials ({len(targets)} targets x {args.trials})")
    if args.out:
        print(f"Writing CSV output to: {args.out}")

    try:
        with CameraStream(cv_threads=1) as stream:
            last_frame_ts = None

            for trial_item in schedule:
                trial_num = trial_item.trial
                target = trial_item.target

                phases = [
                    (PHASE_GET_READY, PHASE_DURATIONS[PHASE_GET_READY], f"[{trial_num}/{total_trials}] GET READY: {target}\a"),
                    (PHASE_SETTLE, PHASE_DURATIONS[PHASE_SETTLE], f"[{trial_num}/{total_trials}] SETTLE\a"),
                    (PHASE_HOLD, PHASE_DURATIONS[PHASE_HOLD], f"[{trial_num}/{total_trials}] HOLD: {target}\a"),
                    (PHASE_RELAX, PHASE_DURATIONS[PHASE_RELAX], f"[{trial_num}/{total_trials}] RELAX\a"),
                ]

                for phase_name, duration, prompt in phases:
                    print(prompt, flush=True)
                    phase_start = time.monotonic()

                    while time.monotonic() - phase_start < duration:
                        if stream.fatal_error:
                            sys.stderr.write(f"Camera fatal error: {stream.fatal_error}\n")
                            return

                        latest = stream.get_latest()
                        if latest is None:
                            time.sleep(0.005)
                            continue

                        frame, frame_ts = latest
                        if last_frame_ts is not None and frame_ts == last_frame_ts:
                            time.sleep(0.005)
                            continue

                        if not limiter.allow():
                            last_frame_ts = frame_ts
                            continue

                        last_frame_ts = frame_ts

                        # Convert frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        ts_ms = mono_ms.next(frame_ts)

                        # Inference
                        t0 = time.perf_counter()
                        result = recognizer.recognize_for_video(mp_image, ts_ms)
                        infer_ms = (time.perf_counter() - t0) * 1000.0

                        state, label, score, handedness = classify(result)

                        # Record telemetry
                        recorded_rows.append(
                            (ts_ms, trial_num, target, phase_name, state, label, score)
                        )
                        phase_infer_times[phase_name].append(infer_ms)
                        total_frames += 1

                        if csv_file is not None:
                            row_prefix = (
                                f"{ts_ms},{trial_num},{target},{phase_name},"
                                f"{state},{label},{score:.4f},{handedness},{infer_ms:.2f}"
                            )
                            if args.landmarks:
                                lm_str = extract_landmarks(result)
                                csv_file.write(f"{row_prefix},{lm_str}\n")
                            else:
                                csv_file.write(f"{row_prefix}\n")

    except KeyboardInterrupt:
        print("\nTally interrupted by user.")
    finally:
        try:
            recognizer.close()
        except Exception:
            pass

        if csv_file is not None:
            try:
                csv_file.close()
            except Exception:
                pass

        total_duration = time.monotonic() - start_monotonic
        print_summary_tables(
            recorded_rows, phase_infer_times, total_duration, total_frames, proc
        )


if __name__ == "__main__":
    main()
