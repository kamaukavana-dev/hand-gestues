#!/usr/bin/env python3
"""Live hand-gesture preview with skeleton overlay and HUD.

Usage:
    .venv-mp/bin/python preview_live.py [--max-fps N] [--seconds N] [--mirror|--no-mirror]

Constraints:
- Camera opened ONLY through CameraStream.
- No cv2.imwrite, no cv2.VideoWriter, no file writes, no network calls.
- Recognizer and camera released on ALL paths (KeyboardInterrupt included).
- MediaPipe API: mp.tasks.vision / mp.tasks.BaseOptions only.
"""
import argparse
import os
import sys
import time

import cv2
import mediapipe as mp
import psutil

from capture import CameraStream
from gesture_logic import MonotonicMs, RateLimiter, classify
from preview_logic import HAND_CONNECTIONS, clipped, hud_lines, to_pixel

# ---------------------------------------------------------------------------
# Drawing constants
# ---------------------------------------------------------------------------
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_YELLOW = (0, 220, 220)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.6
_FONT_THICK = 1
_LINE_HEIGHT_PX = 24
_GUIDE_MARGIN = 0.05           # Must match preview_logic.clipped margin


def _draw_skeleton(frame, landmarks, w, h, mirror, is_clipped):
    """Draw the 21-landmark hand skeleton on *frame* (in-place)."""
    color = _RED if is_clipped else _GREEN
    for a, b in HAND_CONNECTIONS:
        pa = to_pixel(landmarks[a].x, landmarks[a].y, w, h, mirror)
        pb = to_pixel(landmarks[b].x, landmarks[b].y, w, h, mirror)
        cv2.line(frame, pa, pb, color, 1, cv2.LINE_AA)
    for lm in landmarks:
        pt = to_pixel(lm.x, lm.y, w, h, mirror)
        cv2.circle(frame, pt, 3, color, -1, cv2.LINE_AA)


def _draw_guide_rect(frame, w, h):
    """Draw the 5% guide rectangle (thin yellow line)."""
    margin = int(_GUIDE_MARGIN * min(w, h))
    top_left = (margin, margin)
    bot_right = (w - 1 - margin, h - 1 - margin)
    cv2.rectangle(frame, top_left, bot_right, _YELLOW, 1)


def _draw_hud(frame, lines, w, h):
    """Render HUD text lines in the top-left corner."""
    for i, ln in enumerate(lines):
        y = 20 + i * _LINE_HEIGHT_PX
        cv2.putText(frame, ln, (8, y), _FONT, _FONT_SCALE, _WHITE, _FONT_THICK, cv2.LINE_AA)


def _build_recognizer(model_path):
    """Construct a VIDEO-mode GestureRecognizer using mp.tasks.vision API."""
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
    )
    return GestureRecognizer.create_from_options(options)


def main():
    parser = argparse.ArgumentParser(
        description="Live hand-gesture preview with skeleton overlay and HUD."
    )
    parser.add_argument(
        "--max-fps", type=float, default=15.0,
        help="Maximum recognition rate in frames/second (default: 15)"
    )
    parser.add_argument(
        "--seconds", type=float, default=120.0,
        help="Auto-exit after this many seconds (default: 120)"
    )
    mirror_group = parser.add_mutually_exclusive_group()
    mirror_group.add_argument(
        "--mirror", dest="mirror", action="store_true", default=True,
        help="Mirror display horizontally (default: on)"
    )
    mirror_group.add_argument(
        "--no-mirror", dest="mirror", action="store_false",
        help="Disable horizontal mirror"
    )
    args = parser.parse_args()

    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "gesture_recognizer.task"
    )
    if not os.path.isfile(model_path):
        print(f"ERROR: model not found at {model_path}", file=sys.stderr)
        sys.exit(1)

    proc = psutil.Process()
    print(
        f"Starting preview_live.py | max_fps={args.max_fps} | "
        f"seconds={args.seconds} | mirror={args.mirror}"
    )
    print(f"RSS before recognizer: {proc.memory_info().rss / 1_048_576:.1f} MB")

    recognizer = _build_recognizer(model_path)
    print(f"RSS after recognizer:  {proc.memory_info().rss / 1_048_576:.1f} MB")

    mono_ms = MonotonicMs()
    rate = RateLimiter(max_fps=args.max_fps)
    deadline = time.monotonic() + args.seconds

    cam = CameraStream(cv_threads=1)
    try:
        while time.monotonic() < deadline:
            frame_bgr = cam.read()
            if frame_bgr is None:
                continue

            if not rate.allow():
                # Still show the previous frame if any; just skip inference
                cv2.waitKey(1)
                continue

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Build mp.Image
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=frame_rgb,
            )

            # Strictly-increasing ms timestamp
            ts_ms = mono_ms.next(time.monotonic())

            # Inference
            t0 = time.monotonic()
            result = recognizer.recognize_for_video(mp_image, ts_ms)
            infer_ms = (time.monotonic() - t0) * 1000.0

            state, label, score, handedness = classify(result)

            # Landmarks (may be absent)
            landmarks = None
            if result.hand_landmarks:
                landmarks = result.hand_landmarks[0]

            is_clipped_flag = clipped(landmarks) if landmarks else False

            # --- Drawing (mutate a copy displayed to user) ---
            display = frame_bgr.copy() if args.mirror else frame_bgr
            if args.mirror:
                display = cv2.flip(display, 1)

            _draw_guide_rect(display, w, h)

            if landmarks:
                _draw_skeleton(display, landmarks, w, h, args.mirror, is_clipped_flag)

            det_fps = rate.actual_fps() if hasattr(rate, "actual_fps") else args.max_fps
            cam_fps = cam.fps() if hasattr(cam, "fps") else 0.0

            lines = hud_lines(
                state=state,
                label=label,
                score=score,
                infer_ms=infer_ms,
                det_fps=det_fps,
                cam_fps=cam_fps,
                is_clipped=is_clipped_flag,
            )
            _draw_hud(display, lines, w, h)

            cv2.imshow("Gesture Preview", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or Esc
                print("User quit.")
                break

    except KeyboardInterrupt:
        print("\nKeyboardInterrupt — releasing resources.")
    finally:
        recognizer.close()
        cv2.destroyAllWindows()
        print("Recognizer closed. Windows destroyed.")


if __name__ == "__main__":
    main()
