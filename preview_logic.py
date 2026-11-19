#!/usr/bin/env python3
"""Pure Python helper logic for live preview overlay.
Strictly NO cv2, mediapipe, or camera dependencies.
"""
import math

HAND_CONNECTIONS = (
    # Thumb: 0-1-2-3-4
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    # Index finger: 0-5-6-7-8
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    # Middle finger: 5-9-10-11-12
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    # Ring finger: 9-13-14-15-16
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    # Pinky finger: 13-17-18-19-20 plus palm base 0-17
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
)


def to_pixel(x, y, w, h, mirror=False):
    """Converts normalized coordinates (x, y) in [0.0, 1.0] to integer pixel coordinates (px, py).

    If mirror is True, horizontally flips x for display representation.
    Clamps the coordinates to valid pixel indices [0, w - 1] and [0, h - 1].
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"Width and height must be positive, got w={w}, h={h}")

    x_eff = (1.0 - x) if mirror else x
    px = min(max(0, math.floor(x_eff * w)), w - 1)
    py = min(max(0, math.floor(y * h)), h - 1)
    return (int(px), int(py))


def clipped(landmarks, margin=0.05):
    """Returns True if any landmark coordinate lies outside [margin, 1.0 - margin].

    landmarks: iterable of objects having .x and .y attributes, or (x, y) tuples/lists.
    """
    if not landmarks:
        return False

    low = float(margin)
    high = 1.0 - low

    for lm in landmarks:
        if hasattr(lm, "x") and hasattr(lm, "y"):
            lx, ly = lm.x, lm.y
        else:
            lx, ly = lm[0], lm[1]

        if lx < low or lx > high or ly < low or ly > high:
            return True

    return False


def hud_lines(state, label, score, infer_ms, det_fps, cam_fps, is_clipped):
    """Formats HUD telemetry lines for on-screen display.

    Returns a list of display strings.
    """
    lines = []

    # Line 1: Classification
    if state == "GESTURE":
        lines.append(f"State: {state} | {label} ({score:.2f})")
    elif state == "NONE":
        lines.append(f"State: {state} (Hand detected, no gesture)")
    else:
        lines.append("State: NO_HAND")

    # Line 2: FPS & Latency
    lines.append(
        f"FPS: Det={det_fps:.1f} | Cam={cam_fps:.1f} | Infer={infer_ms:.1f}ms"
    )

    # Line 3: Framing boundary status
    if is_clipped:
        lines.append("WARNING: Hand near/outside guide margin!")
    else:
        lines.append("Framing: Hand well-centered")

    return lines
