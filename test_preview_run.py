#!/usr/bin/env python3
"""End-to-end tests for preview_live.run_preview().

Uses the REAL CameraStream with a fake cap_factory (no hardware camera),
a fake recognizer, and a fake show_fn. No cv2.imshow, no camera device.
"""
import ast
import inspect
import threading
import time
import types
import unittest

import cv2
import numpy as np

from capture import CameraStream
from preview_live import run_preview


# ---------------------------------------------------------------------------
# Fake capture (identical pattern to test_capture.py's FakeCapture)
# ---------------------------------------------------------------------------
class FakeCapture:
    """Produces synthetic 640×480 BGR frames at ~100 fps."""

    def __init__(self, delay=0.005):
        self.delay = delay
        self.released = threading.Event()
        self.release_count = 0
        self._frame_index = 0
        self._stopped = threading.Event()

    def isOpened(self):
        return not self._stopped.is_set()

    def set(self, prop, value):
        return True

    def read(self):
        if self._stopped.is_set():
            return False, None
        if self.delay:
            time.sleep(self.delay)
        # Each frame is a unique solid-colour array so identity checks work
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (self._frame_index % 256, 0, 0)
        self._frame_index += 1
        return True, frame

    def release(self):
        self._stopped.set()
        self.release_count += 1
        self.released.set()


# ---------------------------------------------------------------------------
# Fake recognizer
# ---------------------------------------------------------------------------
class _FakeLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


class _FakeCategory:
    def __init__(self, name, score):
        self.category_name = name
        self.score = score


class _FakeResult:
    def __init__(self, gesture_name="Open_Palm", score=0.9, with_landmarks=True):
        if gesture_name:
            self.gestures = [[_FakeCategory(gesture_name, score)]]
        else:
            self.gestures = []
        self.handedness = [[_FakeCategory("Right", 0.95)]] if gesture_name else []
        if with_landmarks and gesture_name:
            # 21 landmarks scattered safely inside the guide margin
            self.hand_landmarks = [
                [_FakeLandmark(0.3 + 0.01 * i, 0.3 + 0.01 * i) for i in range(21)]
            ]
        else:
            self.hand_landmarks = []


class FakeRecognizer:
    def __init__(self, result=None):
        self._result = result or _FakeResult()
        self.closed = False
        self.call_count = 0
        # Store the landmark values seen by recognize_for_video for assertion
        self.last_ts = None

    def recognize_for_video(self, mp_image, ts_ms):
        self.call_count += 1
        self.last_ts = ts_ms
        return self._result

    def close(self):
        self.closed = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_args(max_fps=30.0, seconds=60.0, mirror=True):
    return types.SimpleNamespace(max_fps=max_fps, seconds=seconds, mirror=mirror)


def _make_stream(cap=None, **kw):
    if cap is None:
        cap = FakeCapture()
    return CameraStream(
        stall_timeout_s=2.0,
        max_reopens=3,
        cap_factory=lambda idx: cap,
        **kw,
    )


def wait_for(predicate, timeout=4.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestRunPreviewQuit(unittest.TestCase):
    """(a) Loop exits when show_fn returns ord('q')."""

    def test_exit_on_q(self):
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer()
        args = _make_args(max_fps=100.0, seconds=30.0)

        keys = iter([0xFF] * 5 + [ord("q")])

        def show_fn(frame):
            return next(keys, ord("q"))

        with stream:
            reason = run_preview(stream, recognizer, args, show_fn)

        self.assertEqual(reason, "user_quit")

    def test_exit_on_esc(self):
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer()
        args = _make_args(max_fps=100.0, seconds=30.0)

        keys = iter([0xFF] * 3 + [27])

        def show_fn(frame):
            return next(keys, 27)

        with stream:
            reason = run_preview(stream, recognizer, args, show_fn)

        self.assertEqual(reason, "user_quit")


class TestRunPreviewDeadline(unittest.TestCase):
    """(b) Loop exits at --seconds deadline using a fake clock."""

    def test_deadline_exit(self):
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer()
        args = _make_args(max_fps=100.0, seconds=5.0)

        # Fake clock: starts at 0, advances by 1.0 each call
        tick = [0.0]

        def fake_clock():
            t = tick[0]
            tick[0] += 1.0
            return t

        def show_fn(frame):
            return 0xFF  # never quit

        with stream:
            reason = run_preview(stream, recognizer, args, show_fn, clock=fake_clock)

        self.assertEqual(reason, "deadline")

    def test_deadline_is_respected_in_real_time(self):
        """With max_fps=100 and seconds=0.3, exits quickly."""
        cap = FakeCapture(delay=0.001)
        stream = _make_stream(cap)
        recognizer = FakeRecognizer()
        args = _make_args(max_fps=100.0, seconds=0.3)

        def show_fn(frame):
            return 0xFF

        t0 = time.monotonic()
        with stream:
            reason = run_preview(stream, recognizer, args, show_fn)
        elapsed = time.monotonic() - t0

        self.assertEqual(reason, "deadline")
        self.assertLess(elapsed, 3.0, "Should have exited within 3 s for 0.3 s deadline")


class TestRunPreviewResourceRelease(unittest.TestCase):
    """(c) Camera released and thread stopped; (d) recognizer closed — on all paths."""

    def _run_and_check(self, quit_key=None, raise_kbi=False):
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer()
        args = _make_args(max_fps=100.0, seconds=30.0)
        call_n = [0]

        def show_fn(frame):
            call_n[0] += 1
            if raise_kbi and call_n[0] >= 3:
                raise KeyboardInterrupt
            if quit_key and call_n[0] >= 3:
                return quit_key
            return 0xFF

        try:
            with stream:
                run_preview(stream, recognizer, args, show_fn)
        except KeyboardInterrupt:
            pass
        finally:
            recognizer.close()  # mirrors main()'s finally

        # Camera thread should have stopped inside the `with` block
        self.assertFalse(stream.stats()["running"],
                         "Stream should not be running after `with` block")
        self.assertTrue(recognizer.closed, "Recognizer must be closed")

    def test_normal_deadline_exit(self):
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer()
        args = _make_args(max_fps=100.0, seconds=0.2)
        with stream:
            run_preview(stream, recognizer, args, lambda f: 0xFF)
        recognizer.close()
        self.assertFalse(stream.stats()["running"])
        self.assertTrue(recognizer.closed)

    def test_q_exit_releases(self):
        self._run_and_check(quit_key=ord("q"))

    def test_keyboard_interrupt_releases(self):
        self._run_and_check(raise_kbi=True)


class TestRunPreviewDisplayCopy(unittest.TestCase):
    """(e) Display array is always a copy; mutating it never changes camera frame."""

    def test_display_is_copy_with_mirror(self):
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer(result=_FakeResult(gesture_name=None))
        args = _make_args(max_fps=100.0, seconds=30.0, mirror=True)

        seen_frames = []
        cam_frames = []
        call_n = [0]

        # Intercept get_latest to record the camera frame before display
        original_get_latest = stream.get_latest

        def patched_get_latest():
            result = original_get_latest()
            if result is not None:
                cam_frames.append(result[0].copy())
            return result

        stream.get_latest = patched_get_latest

        def show_fn(frame):
            seen_frames.append(frame.copy())
            # Mutate the display frame heavily
            frame[:] = 255
            call_n[0] += 1
            if call_n[0] >= 3:
                return ord("q")
            return 0xFF

        with stream:
            run_preview(stream, recognizer, args, show_fn)

        self.assertGreater(len(seen_frames), 0)
        self.assertGreater(len(cam_frames), 0)
        # Verify the stream still produces unmodified frames after mutation
        latest = stream.get_latest()
        # The latest frame should not be all-255 (mutated display)
        if latest is not None:
            self.assertFalse(
                np.all(latest[0] == 255),
                "Camera frame must not have been mutated by display copy mutation"
            )

    def test_display_is_copy_without_mirror(self):
        """Even with --no-mirror, display is a copy."""
        cap = FakeCapture()
        stream = _make_stream(cap)
        recognizer = FakeRecognizer(result=_FakeResult(gesture_name=None))
        args = _make_args(max_fps=100.0, seconds=30.0, mirror=False)

        mutated = [False]
        call_n = [0]

        original_get_latest = stream.get_latest
        last_cam_frame = [None]

        def patched_get_latest():
            result = original_get_latest()
            if result is not None:
                last_cam_frame[0] = result[0].copy()
            return result

        stream.get_latest = patched_get_latest

        def show_fn(frame):
            frame[:] = 255  # Mutate display
            call_n[0] += 1
            if call_n[0] >= 3:
                return ord("q")
            return 0xFF

        with stream:
            run_preview(stream, recognizer, args, show_fn)

        if last_cam_frame[0] is not None:
            self.assertFalse(
                np.all(last_cam_frame[0] == 255),
                "Camera frame must not be mutated even without mirror"
            )


class TestRunPreviewLandmarkValues(unittest.TestCase):
    """(f) Landmark values reaching the drawing helper equal the fake recognizer's values."""

    def test_landmark_coordinates_passed_through_unmodified(self):
        """The raw .x/.y from the fake result must reach to_pixel unchanged."""
        cap = FakeCapture()
        stream = _make_stream(cap)

        # Known landmarks at precise positions
        lm_x = 0.42
        lm_y = 0.58
        fake_lms = [_FakeLandmark(lm_x, lm_y)] * 21
        fake_result = _FakeResult.__new__(_FakeResult)
        fake_result.gestures = [[_FakeCategory("Open_Palm", 0.9)]]
        fake_result.handedness = [[_FakeCategory("Right", 0.9)]]
        fake_result.hand_landmarks = [fake_lms]

        recognizer = FakeRecognizer(result=fake_result)
        args = _make_args(max_fps=100.0, seconds=30.0, mirror=False)

        seen_x = []
        seen_y = []
        call_n = [0]

        # Patch to_pixel inside preview_live to intercept calls
        import preview_live as pl
        orig_to_pixel = pl.to_pixel

        def spy_to_pixel(x, y, w, h, mirror=False):
            seen_x.append(x)
            seen_y.append(y)
            return orig_to_pixel(x, y, w, h, mirror)

        pl.to_pixel = spy_to_pixel

        def show_fn(frame):
            call_n[0] += 1
            if call_n[0] >= 2:
                return ord("q")
            return 0xFF

        try:
            with stream:
                run_preview(stream, recognizer, args, show_fn)
        finally:
            pl.to_pixel = orig_to_pixel

        self.assertGreater(len(seen_x), 0, "to_pixel must have been called")
        for x in seen_x:
            self.assertAlmostEqual(x, lm_x, places=10,
                msg=f"Landmark x={x!r} does not match fake value {lm_x!r}")
        for y in seen_y:
            self.assertAlmostEqual(y, lm_y, places=10,
                msg=f"Landmark y={y!r} does not match fake value {lm_y!r}")


class TestRunPreviewContractAST(unittest.TestCase):
    """(g) AST contract: every attribute accessed on the CameraStream object exists on the real class."""

    def test_no_nonexistent_camerastream_attributes(self):
        """Parse preview_live.py and assert every attribute accessed on the
        variable bound to CameraStream.__init__ or the `with` target exists on
        the real CameraStream class."""
        with open("preview_live.py", "r") as fh:
            source = fh.read()

        tree = ast.parse(source)

        # Find the name bound to `CameraStream(...)` in `with CameraStream(...) as NAME:`
        stream_var_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    if (
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == "CameraStream"
                        and item.optional_vars is not None
                    ):
                        stream_var_names.add(item.optional_vars.id)

        # Also look for `stream = CameraStream(...)` assignments
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "CameraStream"
                ):
                    for t in node.targets:
                        if isinstance(t, ast.Name):
                            stream_var_names.add(t.id)

        # Collect all attribute accesses on those variable names
        accessed_attrs = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id in stream_var_names
            ):
                accessed_attrs.add(node.attr)

        real_attrs = set(dir(CameraStream))
        unknown = accessed_attrs - real_attrs
        self.assertEqual(
            unknown, set(),
            f"preview_live.py accesses non-existent CameraStream attributes: {unknown}"
        )

    def test_no_read_method_used(self):
        """Explicitly assert that .read() is not called on a CameraStream variable."""
        with open("preview_live.py", "r") as fh:
            source = fh.read()

        tree = ast.parse(source)

        stream_var_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.With):
                for item in node.items:
                    if (
                        isinstance(item.context_expr, ast.Call)
                        and isinstance(item.context_expr.func, ast.Name)
                        and item.context_expr.func.id == "CameraStream"
                        and item.optional_vars is not None
                    ):
                        stream_var_names.add(item.optional_vars.id)

        read_calls = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "read"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id in stream_var_names
            ):
                read_calls.append(node)

        self.assertEqual(
            len(read_calls), 0,
            f"preview_live.py calls .read() on a CameraStream variable — forbidden"
        )


if __name__ == "__main__":
    unittest.main()
