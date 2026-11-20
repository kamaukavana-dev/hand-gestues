#!/usr/bin/env python3
"""Unit tests for preview_logic.py.

Strictly stdlib unittest — no cv2, no mediapipe, no camera imports.
"""
import unittest
from preview_logic import HAND_CONNECTIONS, clipped, hud_lines, to_pixel


# ---------------------------------------------------------------------------
# Minimal stand-in for a landmark object (has .x and .y)
# ---------------------------------------------------------------------------
class _LM:
    def __init__(self, x, y):
        self.x = x
        self.y = y


# ---------------------------------------------------------------------------
# to_pixel
# ---------------------------------------------------------------------------
class TestToPixel(unittest.TestCase):

    # Basic centre, no mirror
    def test_centre_no_mirror(self):
        self.assertEqual(to_pixel(0.5, 0.5, 640, 480), (320, 240))

    # Top-left corner
    def test_origin_no_mirror(self):
        self.assertEqual(to_pixel(0.0, 0.0, 640, 480), (0, 0))

    # Bottom-right corner: floor(1.0 * 640) = 640, clamped to 639
    def test_far_corner_no_mirror(self):
        self.assertEqual(to_pixel(1.0, 1.0, 640, 480), (639, 479))

    # Mirror: x is flipped → (1 - 0.3) * 640 = 448
    def test_mirror_x_flipped(self):
        self.assertEqual(to_pixel(0.3, 0.5, 640, 480, mirror=True), (448, 240))

    # Mirror at x=0.0 → (1 - 0.0) * 640 = 640, clamped to 639
    def test_mirror_origin_clamped(self):
        self.assertEqual(to_pixel(0.0, 0.0, 640, 480, mirror=True), (639, 0))

    # Mirror at x=1.0 → (1 - 1.0) * 640 = 0
    def test_mirror_far_x(self):
        self.assertEqual(to_pixel(1.0, 0.0, 640, 480, mirror=True), (0, 0))

    # Out-of-range values: negative x clamped to 0
    def test_clamp_negative_x(self):
        px, py = to_pixel(-0.5, 0.5, 640, 480)
        self.assertEqual(px, 0)

    # Out-of-range values: x > 1.0 clamped to w-1
    def test_clamp_overflow_x(self):
        px, py = to_pixel(2.0, 0.5, 640, 480)
        self.assertEqual(px, 639)

    # Out-of-range values: negative y clamped to 0
    def test_clamp_negative_y(self):
        px, py = to_pixel(0.5, -1.0, 640, 480)
        self.assertEqual(py, 0)

    # Out-of-range values: y > 1.0 clamped to h-1
    def test_clamp_overflow_y(self):
        px, py = to_pixel(0.5, 3.0, 640, 480)
        self.assertEqual(py, 479)

    # Result is always a 2-tuple of ints
    def test_returns_int_tuple(self):
        px, py = to_pixel(0.5, 0.5, 640, 480)
        self.assertIsInstance(px, int)
        self.assertIsInstance(py, int)

    # Non-square frame
    def test_non_square_frame(self):
        self.assertEqual(to_pixel(0.5, 0.5, 100, 200), (50, 100))

    # Invalid dimensions raise ValueError
    def test_invalid_zero_width(self):
        with self.assertRaises(ValueError):
            to_pixel(0.5, 0.5, 0, 480)

    def test_invalid_negative_height(self):
        with self.assertRaises(ValueError):
            to_pixel(0.5, 0.5, 640, -1)


# ---------------------------------------------------------------------------
# clipped
# ---------------------------------------------------------------------------
class TestClipped(unittest.TestCase):

    # Empty landmarks → False
    def test_empty_list(self):
        self.assertFalse(clipped([]))

    # All landmarks well inside [0.05, 0.95]
    def test_all_inside(self):
        lms = [_LM(0.1, 0.1), _LM(0.5, 0.5), _LM(0.9, 0.9)]
        self.assertFalse(clipped(lms))

    # x exactly at margin (0.05) → NOT clipped (boundary is exclusive: < low)
    def test_x_exactly_at_margin(self):
        self.assertFalse(clipped([_LM(0.05, 0.5)]))

    # x exactly at high margin (0.95) → NOT clipped
    def test_x_exactly_at_high_margin(self):
        self.assertFalse(clipped([_LM(0.95, 0.5)]))

    # x just below low margin → clipped
    def test_x_just_below_margin(self):
        self.assertTrue(clipped([_LM(0.0499, 0.5)]))

    # x just above high margin → clipped
    def test_x_just_above_high_margin(self):
        self.assertTrue(clipped([_LM(0.9501, 0.5)]))

    # y below margin → clipped
    def test_y_below_margin(self):
        self.assertTrue(clipped([_LM(0.5, 0.03)]))

    # y above high margin → clipped
    def test_y_above_high_margin(self):
        self.assertTrue(clipped([_LM(0.5, 0.97)]))

    # Tuple-based landmarks (x, y) also accepted
    def test_tuple_landmarks(self):
        self.assertFalse(clipped([(0.5, 0.5), (0.6, 0.6)]))
        self.assertTrue(clipped([(0.5, 0.5), (0.03, 0.5)]))

    # Custom margin
    def test_custom_margin(self):
        # margin=0.1 → low=0.1; x=0.1 is exactly at boundary → not clipped
        self.assertFalse(clipped([_LM(0.1, 0.5)], margin=0.1))
        # x=0.09 < 0.1 → clipped
        self.assertTrue(clipped([_LM(0.09, 0.5)], margin=0.1))

    # Mixed: one landmark inside, one outside
    def test_mixed_one_outside(self):
        self.assertTrue(clipped([_LM(0.5, 0.5), _LM(0.01, 0.5)]))


# ---------------------------------------------------------------------------
# HAND_CONNECTIONS skeleton integrity
# ---------------------------------------------------------------------------
class TestHandConnections(unittest.TestCase):

    def test_all_indices_valid(self):
        """Every index in HAND_CONNECTIONS must be in [0, 20]."""
        for a, b in HAND_CONNECTIONS:
            self.assertGreaterEqual(a, 0, msg=f"Index {a} out of range in connection ({a},{b})")
            self.assertLessEqual(a, 20, msg=f"Index {a} out of range in connection ({a},{b})")
            self.assertGreaterEqual(b, 0, msg=f"Index {b} out of range in connection ({a},{b})")
            self.assertLessEqual(b, 20, msg=f"Index {b} out of range in connection ({a},{b})")

    def test_no_self_connections(self):
        """No connection should loop back to itself."""
        for a, b in HAND_CONNECTIONS:
            self.assertNotEqual(a, b, msg=f"Self-connection at index {a}")

    def test_nonzero_connection_count(self):
        self.assertGreater(len(HAND_CONNECTIONS), 0)

    def test_palm_base_connection_present(self):
        """Connection (0, 17) must be present (palm base → pinky base)."""
        self.assertIn((0, 17), HAND_CONNECTIONS)


# ---------------------------------------------------------------------------
# hud_lines
# ---------------------------------------------------------------------------
class TestHudLines(unittest.TestCase):

    def _lines(self, **kwargs):
        defaults = dict(
            state="NO_HAND",
            label="",
            score=0.0,
            infer_ms=50.0,
            det_fps=15.0,
            cam_fps=30.0,
            is_clipped=False,
        )
        defaults.update(kwargs)
        return hud_lines(**defaults)

    # Must always return a list
    def test_returns_list(self):
        result = self._lines()
        self.assertIsInstance(result, list)

    # Must have at least one line
    def test_minimum_one_line(self):
        result = self._lines()
        self.assertGreater(len(result), 0)

    # NO_HAND state
    def test_no_hand_state(self):
        lines = self._lines(state="NO_HAND")
        self.assertTrue(
            any("NO_HAND" in ln for ln in lines),
            msg=f"'NO_HAND' not found in {lines}"
        )

    # NONE state
    def test_none_state(self):
        lines = self._lines(state="NONE")
        self.assertTrue(
            any("NONE" in ln for ln in lines),
            msg=f"'NONE' not found in {lines}"
        )

    # GESTURE state: must include label and formatted score
    def test_gesture_state_label_and_score(self):
        lines = self._lines(state="GESTURE", label="Open_Palm", score=0.93)
        first = lines[0]
        self.assertIn("GESTURE", first)
        self.assertIn("Open_Palm", first)
        # Score formatted to 2 d.p.
        self.assertIn("0.93", first)

    # Clipped warning
    def test_clipped_warning_present(self):
        lines = self._lines(is_clipped=True)
        self.assertTrue(
            any("WARNING" in ln for ln in lines),
            msg=f"'WARNING' not found in {lines}"
        )

    # No clipped warning when False
    def test_no_warning_when_not_clipped(self):
        lines = self._lines(is_clipped=False)
        self.assertFalse(
            any("WARNING" in ln for ln in lines),
            msg=f"Unexpected 'WARNING' in {lines}"
        )

    # FPS / latency line must contain numeric values
    def test_fps_latency_line_present(self):
        lines = self._lines(det_fps=12.3, cam_fps=29.9, infer_ms=66.5)
        combined = " ".join(lines)
        self.assertIn("12.3", combined)
        self.assertIn("29.9", combined)
        self.assertIn("66.5", combined)

    # All returned lines are strings
    def test_all_lines_are_strings(self):
        for ln in self._lines(state="GESTURE", label="Victory", score=0.8, is_clipped=True):
            self.assertIsInstance(ln, str)


if __name__ == "__main__":
    unittest.main()
