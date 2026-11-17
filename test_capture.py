#!/usr/bin/env python3
import threading
import time
import unittest

import cv2

from capture import CameraStream


class FakeCapture:
    def __init__(self, block_event=None, error=None, delay=0.005):
        self.block_event = block_event
        self.error = error
        self.delay = delay
        self.released = threading.Event()
        self.release_count = 0
        self.read_count = 0
        self._next_frame = 0

    def isOpened(self):
        return True

    def set(self, prop, value):
        return True

    def read(self):
        self.read_count += 1
        if self.block_event is not None:
            self.block_event.wait()
        if self.error is not None:
            raise self.error
        if self.delay:
            time.sleep(self.delay)
        frame = self._next_frame
        self._next_frame += 1
        return True, frame

    def release(self):
        self.release_count += 1
        self.released.set()


class CameraStreamTests(unittest.TestCase):
    def wait_for(self, predicate, timeout=3.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(0.01)
        return False

    def test_construction_sets_cv_threads_to_one(self):
        CameraStream(cap_factory=lambda index: FakeCapture())
        self.assertEqual(cv2.getNumThreads(), 1)

    def test_latest_frame_only(self):
        cap = FakeCapture(delay=0.01)
        stream = CameraStream(
            stall_timeout_s=1.0,
            max_reopens=3,
            cap_factory=lambda index: cap,
        )
        self.addCleanup(stream.stop)
        stream.start()

        self.assertTrue(
            self.wait_for(lambda: stream.stats()["frames_captured"] >= 10)
        )
        first = stream.get_latest()
        self.assertIsNotNone(first)
        time.sleep(0.05)

        started = time.monotonic()
        second = stream.get_latest()
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 0.1)
        self.assertIsNotNone(second)
        self.assertGreater(second[0], first[0])
        self.assertGreaterEqual(second[1], first[1])

        stats = stream.stats()
        self.assertGreaterEqual(second[0], stats["frames_captured"] - 2)

    def test_stalled_read_triggers_release_and_reopen(self):
        first_event = threading.Event()
        first = FakeCapture(block_event=first_event)
        second = FakeCapture(delay=0.01)
        factory_calls = []
        second_created = threading.Event()

        def factory(index):
            factory_calls.append(index)
            if len(factory_calls) == 1:
                return first
            second_created.set()
            return second

        stream = CameraStream(
            stall_timeout_s=0.2,
            max_reopens=3,
            cap_factory=factory,
        )
        self.addCleanup(stream.stop)
        self.addCleanup(first_event.set)
        stream.start()

        self.assertTrue(first.released.wait(2.0))
        self.assertTrue(second_created.wait(2.0))
        self.assertTrue(
            self.wait_for(lambda: stream.stats()["frames_captured"] >= 1)
        )
        stats = stream.stats()
        self.assertGreaterEqual(stats["stall_count"], 1)
        self.assertGreaterEqual(stats["reopen_count"], 1)
        self.assertIsNone(stats["fatal_error"])

    def test_max_reopens_respected(self):
        events = []

        def factory(index):
            event = threading.Event()
            events.append(event)
            return FakeCapture(block_event=event)

        stream = CameraStream(
            stall_timeout_s=0.1,
            max_reopens=2,
            cap_factory=factory,
        )
        self.addCleanup(stream.stop)
        self.addCleanup(lambda: [event.set() for event in events])
        stream.start()

        self.assertTrue(
            self.wait_for(
                lambda: stream.stats()["fatal_error"] is not None,
                timeout=6.0,
            )
        )
        stats = stream.stats()
        self.assertIsNotNone(stats["fatal_error"])
        self.assertEqual(stats["reopen_count"], 2)
        self.assertFalse(stats["running"])

    def test_stop_releases_and_thread_exits(self):
        cap = FakeCapture(delay=0.005)
        stream = CameraStream(
            stall_timeout_s=1.0,
            max_reopens=3,
            cap_factory=lambda index: cap,
        )
        self.addCleanup(stream.stop)
        stream.start()

        self.assertTrue(
            self.wait_for(lambda: stream.stats()["frames_captured"] >= 1)
        )
        thread = stream._thread
        stream.stop()

        self.assertTrue(cap.released.is_set())
        self.assertFalse(thread.is_alive())
        self.assertFalse(stream.stats()["running"])

    def test_stop_after_error_still_releases(self):
        caps = []

        def factory(index):
            cap = FakeCapture(error=RuntimeError("boom"))
            caps.append(cap)
            return cap

        stream = CameraStream(
            stall_timeout_s=0.5,
            max_reopens=3,
            cap_factory=factory,
        )
        self.addCleanup(stream.stop)
        stream.start()

        self.assertTrue(
            self.wait_for(
                lambda: bool(caps) and caps[0].released.is_set(),
                timeout=2.0,
            )
        )
        stream.stop()

        self.assertTrue(caps[0].released.is_set())
        self.assertFalse(stream.stats()["running"])


if __name__ == "__main__":
    unittest.main()
