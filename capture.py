#!/usr/bin/env python3
import threading
import time
from collections import deque

import cv2


class CameraStream:
    def __init__(
        self,
        index=0,
        width=640,
        height=480,
        stall_timeout_s=3.0,
        max_reopens=3,
        cap_factory=None,
    ):
        self.index = index
        self.width = width
        self.height = height
        self.stall_timeout_s = stall_timeout_s
        self.max_reopens = max_reopens
        self.cap_factory = cap_factory or (
            lambda camera_index: cv2.VideoCapture(camera_index)
        )

        self._condition = threading.Condition()
        self._cap_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._cap = None
        self._latest = None
        self._timestamps = deque(maxlen=60)
        self._frames_captured = 0
        self._stall_count = 0
        self._reopen_count = 0
        self._consecutive_failures = 0
        self._fatal_error = None
        self._running = False
        self._generation = 0
        self._read_error = None
        self._last_frame_time = None

    def _measured_fps_locked(self):
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self._timestamps) - 1) / elapsed

    def get_latest(self):
        with self._condition:
            return self._latest

    def stats(self):
        with self._condition:
            return {
                "fps": self._measured_fps_locked(),
                "measured_fps": self._measured_fps_locked(),
                "frames_captured": self._frames_captured,
                "stall_count": self._stall_count,
                "reopen_count": self._reopen_count,
                "fatal_error": self._fatal_error,
                "running": self._running,
            }

    @property
    def fatal_error(self):
        with self._condition:
            return self._fatal_error

    def start(self):
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._latest = None
            self._timestamps.clear()
            self._frames_captured = 0
            self._stall_count = 0
            self._reopen_count = 0
            self._consecutive_failures = 0
            self._fatal_error = None
            self._read_error = None
            self._last_frame_time = None
            self._running = True
            self._thread = threading.Thread(
                target=self._run,
                daemon=True,
                name="CameraStream",
            )
            self._thread.start()
        return self

    def stop(self):
        self._stop_event.set()
        with self._condition:
            self._generation += 1
            self._condition.notify_all()
        self._release_current_cap()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.stall_timeout_s + 2.0))
        with self._condition:
            self._running = False
            self._condition.notify_all()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
        return False

    def _open_cap(self):
        try:
            cap = self.cap_factory(self.index)
        except Exception:
            return None
        if cap is None:
            return None
        try:
            is_open = cap.isOpened() if hasattr(cap, "isOpened") else True
            if not is_open:
                cap.release()
                return None
            if hasattr(cap, "set"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except Exception:
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap

    def _release_current_cap(self):
        with self._cap_lock:
            cap = self._cap
            self._cap = None
        if cap is None:
            return
        try:
            cap.release()
        except Exception:
            pass
        print("CAMERA RELEASED")

    def _reader_loop(self, cap, generation):
        while not self._stop_event.is_set():
            with self._condition:
                if generation != self._generation:
                    return
            try:
                ok, frame = cap.read()
            except Exception as exc:
                with self._condition:
                    if generation == self._generation and not self._stop_event.is_set():
                        self._read_error = f"read error: {exc}"
                        self._condition.notify_all()
                return
            if not ok:
                with self._condition:
                    if generation == self._generation and not self._stop_event.is_set():
                        self._read_error = "read returned False"
                        self._condition.notify_all()
                return
            now = time.monotonic()
            with self._condition:
                if generation != self._generation or self._stop_event.is_set():
                    return
                self._latest = (frame, now)
                self._timestamps.append(now)
                self._frames_captured += 1
                self._last_frame_time = now
                self._consecutive_failures = 0
                self._condition.notify_all()

    def _wait_for_activity(self, generation):
        last_seen = time.monotonic()
        deadline = last_seen + self.stall_timeout_s
        with self._condition:
            while not self._stop_event.is_set():
                if generation != self._generation:
                    return "abandoned", None
                if self._read_error is not None:
                    error = self._read_error
                    self._read_error = None
                    return "error", error
                if (
                    self._last_frame_time is not None
                    and self._last_frame_time > last_seen
                ):
                    last_seen = self._last_frame_time
                    deadline = last_seen + self.stall_timeout_s
                    continue
                now = time.monotonic()
                if now >= deadline:
                    return "stall", None
                self._condition.wait(timeout=deadline - now)
        return "stopped", None

    def _handle_failure(self, message):
        with self._condition:
            self._consecutive_failures += 1
            if self._consecutive_failures > self.max_reopens:
                self._fatal_error = (
                    f"{message}; max_reopens={self.max_reopens} exceeded"
                )
                self._condition.notify_all()
                return False
        if self._stop_event.wait(1.0):
            return False
        return True

    def _abandon_generation(self):
        with self._condition:
            self._generation += 1
            self._condition.notify_all()

    def _run(self):
        first_attempt = True
        try:
            while not self._stop_event.is_set():
                if not first_attempt:
                    with self._condition:
                        self._reopen_count += 1
                first_attempt = False

                cap = self._open_cap()
                if self._stop_event.is_set():
                    if cap is not None:
                        try:
                            cap.release()
                        except Exception:
                            pass
                    break
                if cap is None:
                    if not self._handle_failure("camera open failed"):
                        break
                    continue

                with self._cap_lock:
                    self._cap = cap
                print("CAMERA LIVE")
                with self._condition:
                    self._generation += 1
                    generation = self._generation
                    self._read_error = None
                    self._condition.notify_all()

                worker = threading.Thread(
                    target=self._reader_loop,
                    args=(cap, generation),
                    daemon=True,
                    name=f"CameraStreamReader-{generation}",
                )
                worker.start()

                outcome, error = self._wait_for_activity(generation)
                if self._stop_event.is_set():
                    break
                if outcome == "stall":
                    with self._condition:
                        self._stall_count += 1
                    self._abandon_generation()
                    self._release_current_cap()
                    if not self._handle_failure(
                        f"no frame for {self.stall_timeout_s:g}s"
                    ):
                        break
                elif outcome == "error":
                    self._abandon_generation()
                    self._release_current_cap()
                    if not self._handle_failure(error or "read error"):
                        break
                elif outcome == "abandoned":
                    self._release_current_cap()
                    if not self._handle_failure("capture generation abandoned"):
                        break
                else:
                    break
        finally:
            self._release_current_cap()
            with self._condition:
                self._running = False
                self._condition.notify_all()
