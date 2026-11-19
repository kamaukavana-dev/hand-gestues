#!/usr/bin/env python3
"""Unit tests for gesture recognition business logic, rate limiter, and telemetry stats."""
import unittest

from gesture_logic import RateLimiter, WindowStats, classify


class FakeCategory:
    def __init__(self, category_name, score):
        self.category_name = category_name
        self.score = score


class FakeResult:
    def __init__(self, gestures=None, handedness=None):
        self.gestures = gestures if gestures is not None else []
        self.handedness = handedness if handedness is not None else []


class FakeClock:
    def __init__(self, start_time=0.0):
        self.current_time = float(start_time)

    def __call__(self):
        return self.current_time

    def advance(self, seconds):
        self.current_time += float(seconds)


class TestGestureLogic(unittest.TestCase):
    def test_empty_result_returns_no_hand(self):
        # Empty gestures list
        state, label, score, handedness = classify(FakeResult([]))
        self.assertEqual(state, "NO_HAND")
        self.assertEqual(score, 0.0)

        # None result
        state, label, score, handedness = classify(None)
        self.assertEqual(state, "NO_HAND")

        # Empty inner list
        state, label, score, handedness = classify(FakeResult([[]]))
        self.assertEqual(state, "NO_HAND")

    def test_none_category_returns_none_state(self):
        res = FakeResult(
            gestures=[[FakeCategory("None", 0.85)]],
            handedness=[[FakeCategory("Right", 0.99)]],
        )
        state, label, score, handedness = classify(res)
        self.assertEqual(state, "NONE")
        self.assertEqual(label, "None")
        self.assertAlmostEqual(score, 0.85)
        self.assertEqual(handedness, "Right")

    def test_open_palm_gesture_and_handedness(self):
        res = FakeResult(
            gestures=[[FakeCategory("Open_Palm", 0.93), FakeCategory("Closed_Fist", 0.05)]],
            handedness=[[FakeCategory("Left", 0.97)]],
        )
        state, label, score, handedness = classify(res)
        self.assertEqual(state, "GESTURE")
        self.assertEqual(label, "Open_Palm")
        self.assertAlmostEqual(score, 0.93)
        self.assertEqual(handedness, "Left")


class TestRateLimiter(unittest.TestCase):
    def test_never_exceeds_max_fps_over_10_seconds(self):
        max_fps = 8.0
        clock = FakeClock(100.0)
        limiter = RateLimiter(max_fps=max_fps, clock=clock)

        # Step by 10ms for 10 simulated seconds (1,000 steps)
        allowed_count = 0
        step_s = 0.010
        total_steps = int(10.0 / step_s)

        for _ in range(total_steps):
            if limiter.allow():
                allowed_count += 1
            clock.advance(step_s)

        # Over 10s at 8 FPS, total calls allowed cannot exceed 81 (1 at t=0 + 80 intervals)
        self.assertLessEqual(allowed_count, int(10.0 * max_fps) + 1)
        self.assertGreaterEqual(allowed_count, int(10.0 * max_fps) - 1)

    def test_no_burst_after_long_stall(self):
        max_fps = 8.0  # interval = 125ms
        clock = FakeClock(10.0)
        limiter = RateLimiter(max_fps=max_fps, clock=clock)

        # First call allowed
        self.assertTrue(limiter.allow())

        # Advance normally 1 interval
        clock.advance(0.125)
        self.assertTrue(limiter.allow())

        # Long stall: 10 seconds of no calls
        clock.advance(10.0)

        # First call after stall should be allowed (resets deadline)
        self.assertTrue(limiter.allow())

        # Consecutive immediate calls within the same interval must all be rejected (NO BURST)
        for _ in range(50):
            clock.advance(0.001)  # 1ms step
            self.assertFalse(limiter.allow())


class TestWindowStats(unittest.TestCase):
    def test_known_p95_and_stats(self):
        stats = WindowStats()
        # Add 100 known values: 0 to 99
        for i in range(100):
            stats.add(infer_ms=float(i), convert_ms=1.5, state="GESTURE", label="Victory")

        self.assertEqual(stats.count(), 100)
        self.assertAlmostEqual(stats.mean(stats.infer_times), 49.5)
        # 95th percentile of 0..99 is 95.0
        self.assertAlmostEqual(stats.p95(stats.infer_times), 95.0)

    def test_state_fractions(self):
        stats = WindowStats()
        for _ in range(60):
            stats.add(50.0, 1.0, "NO_HAND", "")
        for _ in range(30):
            stats.add(60.0, 1.2, "GESTURE", "Open_Palm")
        for _ in range(10):
            stats.add(55.0, 1.1, "NONE", "None")

        fractions = stats.state_fractions()
        self.assertAlmostEqual(fractions["NO_HAND"], 0.60)
        self.assertAlmostEqual(fractions["GESTURE"], 0.30)
        self.assertAlmostEqual(fractions["NONE"], 0.10)


if __name__ == "__main__":
    unittest.main()
