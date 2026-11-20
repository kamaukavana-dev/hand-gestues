#!/usr/bin/env python3
"""Unit tests for ground-truth tally logic, scheduling, scoring, and debounce simulation."""
from collections import Counter
import unittest

from tally_logic import (
    DEFAULT_TARGETS,
    PHASE_HOLD,
    PHASE_GET_READY,
    PHASE_RELAX,
    build_schedule,
    debounce_sim,
    debounce_sim_multi,
    score_trials,
    summarize,
)


class TestTallySchedule(unittest.TestCase):
    def test_schedule_balanced_and_seed_deterministic(self):
        trials = 8
        s1 = build_schedule(trials=trials, seed=123)
        s2 = build_schedule(trials=trials, seed=123)
        s3 = build_schedule(trials=trials, seed=456)

        total_expected = len(DEFAULT_TARGETS) * trials
        self.assertEqual(len(s1), total_expected)
        self.assertEqual(len(s2), total_expected)

        # Balanced: each target appears exactly `trials` times
        counts = Counter(item.target for item in s1)
        for t in DEFAULT_TARGETS:
            self.assertEqual(counts[t], trials)

        # Deterministic with identical seed
        self.assertEqual([x.target for x in s1], [x.target for x in s2])

        # Different seed produces different shuffle
        self.assertNotEqual([x.target for x in s1], [x.target for x in s3])


class TestTallyScoring(unittest.TestCase):
    def test_scoring_and_first_correct_ms(self):
        # 2 trials:
        # Trial 1 (Open_Palm): 10 HOLD frames; first 2 None, next 8 Open_Palm @ score 0.92
        # Trial 2 (REST): 10 HOLD frames; 8 NO_HAND, 2 None
        rows = []
        t = 1000

        # Trial 1: Open_Palm
        for i in range(10):
            t += 60
            if i < 2:
                rows.append((t, 1, "Open_Palm", PHASE_HOLD, "NONE", "None", 0.5))
            else:
                rows.append((t, 1, "Open_Palm", PHASE_HOLD, "GESTURE", "Open_Palm", 0.92))

        # Trial 2: REST
        for i in range(10):
            t += 60
            if i < 8:
                rows.append((t, 2, "REST", PHASE_HOLD, "NO_HAND", "", 0.0))
            else:
                rows.append((t, 2, "REST", PHASE_HOLD, "NONE", "None", 0.4))

        results = score_trials(rows)
        self.assertEqual(len(results), 2)

        r1 = results[0]
        self.assertEqual(r1.trial, 1)
        self.assertEqual(r1.target, "Open_Palm")
        self.assertEqual(r1.hold_frame_count, 10)
        self.assertAlmostEqual(r1.fraction_correct, 0.80)
        self.assertEqual(r1.first_correct_ms, 120)  # frame index 2 -> 2 * 60ms = 120ms
        self.assertAlmostEqual(r1.max_score_correct, 0.92)
        self.assertEqual(r1.predicted_counts["Open_Palm"], 8)
        self.assertEqual(r1.predicted_counts["None"], 2)

        r2 = results[1]
        self.assertEqual(r2.trial, 2)
        self.assertEqual(r2.target, "REST")
        self.assertEqual(r2.hold_frame_count, 10)
        self.assertAlmostEqual(r2.fraction_correct, 1.0)
        self.assertEqual(r2.predicted_counts["NO_HAND"], 8)
        self.assertEqual(r2.predicted_counts["None"], 2)


class TestTallySummarize(unittest.TestCase):
    def test_confusion_matrix_and_threshold_effects(self):
        # Target Victory: 10 frames with varying scores
        # 4 frames Victory 0.95
        # 3 frames Victory 0.65
        # 2 frames Closed_Fist 0.80 (confusion)
        # 1 frame NO_HAND 0.0
        rows = []
        t = 1000
        for _ in range(4):
            t += 60
            rows.append((t, 1, "Victory", PHASE_HOLD, "GESTURE", "Victory", 0.95))
        for _ in range(3):
            t += 60
            rows.append((t, 1, "Victory", PHASE_HOLD, "GESTURE", "Victory", 0.65))
        for _ in range(2):
            t += 60
            rows.append((t, 1, "Victory", PHASE_HOLD, "GESTURE", "Closed_Fist", 0.80))
        for _ in range(1):
            t += 60
            rows.append((t, 1, "Victory", PHASE_HOLD, "NO_HAND", "", 0.0))

        summary = summarize(rows, thresholds=(0.5, 0.7, 0.9))
        matrix = summary["confusion_matrix"]
        self.assertEqual(matrix["Victory"]["Victory"], 7)
        self.assertEqual(matrix["Victory"]["Closed_Fist"], 2)
        self.assertEqual(matrix["Victory"]["NO_HAND"], 1)

        v_summary = summary["targets"]["Victory"]["thresholds"]
        # At thr=0.5: 7/10 Victory frames -> recall 0.70 >= 0.60 -> pass_rate 1.0
        self.assertAlmostEqual(v_summary[0.5]["mean_recall"], 0.70)
        self.assertEqual(v_summary[0.5]["pass_rate"], 1.0)

        # At thr=0.7: only the 4 frames >= 0.7 count -> recall 0.40 < 0.60 -> pass_rate 0.0
        self.assertAlmostEqual(v_summary[0.7]["mean_recall"], 0.40)
        self.assertEqual(v_summary[0.7]["pass_rate"], 0.0)

        # At thr=0.9: 4/10 -> recall 0.40
        self.assertAlmostEqual(v_summary[0.9]["mean_recall"], 0.40)

    def test_rest_pass_and_fail(self):
        # Trial 1: Clean REST (all NO_HAND)
        rows_clean = [(i * 60, 1, "REST", PHASE_HOLD, "NO_HAND", "", 0.0) for i in range(10)]
        s_clean = summarize(rows_clean, thresholds=(0.6,))
        self.assertEqual(s_clean["targets"]["REST"]["thresholds"][0.6]["pass_rate"], 1.0)
        self.assertEqual(s_clean["rest_emissions"][0.6], 0)

        # Trial 2: Noisy REST (one gesture frame with score 0.75)
        rows_noisy = [(i * 60, 2, "REST", PHASE_HOLD, "NO_HAND", "", 0.0) for i in range(9)]
        rows_noisy.append((600, 2, "REST", PHASE_HOLD, "GESTURE", "Thumb_Up", 0.75))

        s_noisy = summarize(rows_noisy, thresholds=(0.6, 0.8))
        # At thr=0.6, Thumb_Up 0.75 >= 0.6 -> fails REST pass
        self.assertEqual(s_noisy["targets"]["REST"]["thresholds"][0.6]["pass_rate"], 0.0)
        self.assertEqual(s_noisy["rest_emissions"][0.6], 1)

        # At thr=0.8, Thumb_Up 0.75 < 0.8 -> passes
        self.assertEqual(s_noisy["targets"]["REST"]["thresholds"][0.8]["pass_rate"], 1.0)
        self.assertEqual(s_noisy["rest_emissions"][0.8], 0)


class TestDebounceSimulation(unittest.TestCase):
    def test_clean_sequence_firing(self):
        # 11 frames spaced 60ms: t=0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600ms
        # Total continuous hold time: 600ms
        rows = [
            (i * 60, 1, "Open_Palm", PHASE_HOLD, "GESTURE", "Open_Palm", 0.90)
            for i in range(11)
        ]
        # hold_ms=250 should fire
        r250 = debounce_sim(rows, hold_ms=250, thr=0.6)
        self.assertEqual(r250["target_fired"], 1)

        # hold_ms=400 should fire
        r400 = debounce_sim(rows, hold_ms=400, thr=0.6)
        self.assertEqual(r400["target_fired"], 1)

        # hold_ms=600 should fire
        r600 = debounce_sim(rows, hold_ms=600, thr=0.6)
        self.assertEqual(r600["target_fired"], 1)

        # hold_ms=800 should NOT fire (duration only 600ms)
        r800 = debounce_sim(rows, hold_ms=800, thr=0.6)
        self.assertEqual(r800["target_fired"], 0)

    def test_flickering_sequence_no_fire(self):
        # Alternating Open_Palm and None every 60ms
        rows = []
        for i in range(20):
            t = i * 60
            if i % 2 == 0:
                rows.append((t, 1, "Open_Palm", PHASE_HOLD, "GESTURE", "Open_Palm", 0.90))
            else:
                rows.append((t, 1, "Open_Palm", PHASE_HOLD, "NONE", "None", 0.50))

        # Because it flickers every 60ms, continuous duration never reaches 250ms
        res = debounce_sim(rows, hold_ms=250, thr=0.6)
        self.assertEqual(res["target_fired"], 0)

    def test_low_score_sequence_no_fire(self):
        # 15 frames of Open_Palm, but score is 0.45 < thr=0.6
        rows = [
            (i * 60, 1, "Open_Palm", PHASE_HOLD, "GESTURE", "Open_Palm", 0.45)
            for i in range(15)
        ]
        res = debounce_sim(rows, hold_ms=250, thr=0.6)
        self.assertEqual(res["target_fired"], 0)


if __name__ == "__main__":
    unittest.main()
