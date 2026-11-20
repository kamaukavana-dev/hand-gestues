#!/usr/bin/env python3
"""Gesture recognition business logic, rate limiting, and telemetry window statistics.
Pure Python: strictly NO mediapipe, cv2, or camera dependencies.
"""
import time


def classify(result):
    """Classify a GestureRecognizerResult-like object.

    Object structure:
      - result.gestures: list[list[Category]]
      - result.handedness: list[list[Category]]
      - Category: category_name (str), score (float)

    Returns:
      (state, label, score, handedness)
      state in ("NO_HAND", "NONE", "GESTURE")
    """
    if result is None:
        return ("NO_HAND", "", 0.0, "")

    gestures = getattr(result, "gestures", None)
    if not gestures or len(gestures) == 0 or len(gestures[0]) == 0:
        return ("NO_HAND", "", 0.0, "")

    first_hand_gestures = gestures[0]
    top_cat = first_hand_gestures[0]
    label = getattr(top_cat, "category_name", "")
    score = float(getattr(top_cat, "score", 0.0))

    handedness = ""
    handedness_list = getattr(result, "handedness", None)
    if handedness_list and len(handedness_list) > 0 and len(handedness_list[0]) > 0:
        handedness = getattr(handedness_list[0][0], "category_name", "")

    if label == "None":
        return ("NONE", label, score, handedness)
    return ("GESTURE", label, score, handedness)


class MonotonicMs:
    """Generates strictly increasing integer millisecond timestamps from seconds.

    next(t_seconds) returns int(t_seconds * 1000). If that is <= last, returns last + 1.
    """

    def __init__(self):
        self.last_ts_ms = None

    def next(self, t_seconds):
        ts = int(float(t_seconds) * 1000)
        if self.last_ts_ms is not None and ts <= self.last_ts_ms:
            ts = self.last_ts_ms + 1
        self.last_ts_ms = ts
        return ts

    def reset(self):
        self.last_ts_ms = None


class RateLimiter:
    """Enforces a maximum execution rate (max_fps).

    Advances the deadline by 1.0 / max_fps each time allow() returns True.
    If the clock falls more than one interval behind the current time (e.g. after
    a stall), resets to now to prevent catch-up bursts.
    """

    def __init__(self, max_fps, clock=time.monotonic):
        if max_fps <= 0:
            raise ValueError(f"max_fps must be positive, got {max_fps}")
        self.max_fps = float(max_fps)
        self.interval = 1.0 / self.max_fps
        self.clock = clock
        self._next_deadline = None

    def allow(self):
        now = self.clock()
        if self._next_deadline is None:
            self._next_deadline = now + self.interval
            return True

        if now < self._next_deadline:
            return False

        # If behind by more than one full interval, reset deadline to avoid catch-up burst
        if (now - self._next_deadline) > self.interval:
            self._next_deadline = now + self.interval
        else:
            self._next_deadline += self.interval
        return True


class WindowStats:
    """Collects inference time, conversion time, states, and labels over a window.

    Calculates sample count, mean, p95 latency, and fractional breakdown of states.
    """

    def __init__(self):
        self.infer_times = []
        self.convert_times = []
        self.states = []
        self.labels = []

    def add(self, infer_ms, convert_ms, state, label):
        self.infer_times.append(float(infer_ms))
        self.convert_times.append(float(convert_ms))
        self.states.append(str(state))
        self.labels.append(str(label))

    def count(self):
        return len(self.infer_times)

    def mean(self, values=None):
        vals = self.infer_times if values is None else values
        if not vals:
            return 0.0
        return float(sum(vals) / len(vals))

    def p95(self, values=None):
        vals = self.infer_times if values is None else values
        if not vals:
            return 0.0
        s = sorted(vals)
        idx = int(len(s) * 0.95)
        if idx >= len(s):
            idx = len(s) - 1
        return float(s[idx])

    def state_fractions(self):
        if not self.states:
            return {}
        total = len(self.states)
        counts = {}
        for s in self.states:
            counts[s] = counts.get(s, 0) + 1
        return {k: v / total for k, v in counts.items()}

    def reset(self):
        self.infer_times.clear()
        self.convert_times.clear()
        self.states.clear()
        self.labels.clear()
