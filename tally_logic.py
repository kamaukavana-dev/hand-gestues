#!/usr/bin/env python3
"""Ground-truth tally logic, scheduling, trial scoring, and debounce simulation.
Pure Python: strictly NO mediapipe, cv2, or camera dependencies.
"""
from collections import Counter, defaultdict, namedtuple
import random

DEFAULT_TARGETS = (
    "Open_Palm",
    "Closed_Fist",
    "Thumb_Up",
    "Victory",
    "Pointing_Up",
    "REST",
)

PHASE_GET_READY = "GET_READY"
PHASE_SETTLE = "SETTLE"
PHASE_HOLD = "HOLD"
PHASE_RELAX = "RELAX"

PHASE_DURATIONS = {
    PHASE_GET_READY: 2.0,
    PHASE_SETTLE: 0.6,
    PHASE_HOLD: 2.5,
    PHASE_RELAX: 1.2,
}

TRIAL_TOTAL_DURATION = sum(PHASE_DURATIONS.values())  # 6.3s

ScheduledTrial = namedtuple("ScheduledTrial", ["trial", "target"])
TrialResult = namedtuple(
    "TrialResult",
    [
        "trial",
        "target",
        "hold_frame_count",
        "predicted_counts",
        "fraction_correct",
        "first_correct_ms",
        "max_score_correct",
    ],
)


def build_schedule(targets=None, trials=8, seed=None):
    """Generates a deterministic, balanced trial schedule.

    Each target in targets is repeated exactly trials times, then shuffled using
    random.Random(seed). Returns a list of ScheduledTrial(trial=1..N, target=name).
    """
    if targets is None:
        target_list = list(DEFAULT_TARGETS)
    else:
        target_list = list(targets)

    if trials <= 0:
        raise ValueError(f"trials must be positive, got {trials}")
    if not target_list:
        raise ValueError("targets list cannot be empty")

    unshuffled = []
    for t in target_list:
        unshuffled.extend([t] * trials)

    rng = random.Random(seed)
    rng.shuffle(unshuffled)

    return [
        ScheduledTrial(trial=idx + 1, target=target)
        for idx, target in enumerate(unshuffled)
    ]


def normalize_row(row):
    """Normalizes dict, namedtuple, or tuple row into a standard tuple:

    (ts_ms, trial, target, phase, state, label, score)
    """
    if isinstance(row, dict):
        return (
            int(row["ts_ms"]),
            int(row["trial"]),
            str(row["target"]),
            str(row["phase"]),
            str(row["state"]),
            str(row.get("label", "")),
            float(row.get("score", 0.0)),
        )
    if hasattr(row, "_fields"):
        return (
            int(row.ts_ms),
            int(row.trial),
            str(row.target),
            str(row.phase),
            str(row.state),
            str(row.label),
            float(row.score),
        )
    return (
        int(row[0]),
        int(row[1]),
        str(row[2]),
        str(row[3]),
        str(row[4]),
        str(row[5]),
        float(row[6]),
    )


def score_trials(rows):
    """Scores HOLD phase rows into per-trial results:

    counts per predicted label (NO_HAND and None kept as labels),
    fraction correct, first_correct_ms, max score of correct frames.
    """
    # Group rows by trial
    by_trial = defaultdict(list)
    trial_target = {}

    for raw in rows:
        ts_ms, trial, target, phase, state, label, score = normalize_row(raw)
        trial_target[trial] = target
        if phase == PHASE_HOLD:
            by_trial[trial].append((ts_ms, target, state, label, score))

    results = []
    for trial in sorted(trial_target.keys()):
        target = trial_target[trial]
        hold_frames = by_trial.get(trial, [])
        predicted_counts = Counter()

        correct_frames = []
        hold_start_ts = hold_frames[0][0] if hold_frames else 0

        for ts_ms, tgt, state, label, score in hold_frames:
            # Normalize effective label for counting
            if state == "NO_HAND":
                eff_label = "NO_HAND"
            elif state == "NONE" or label == "None":
                eff_label = "None"
            else:
                eff_label = label

            predicted_counts[eff_label] += 1

            # Determine correctness
            if target == "REST":
                is_correct = (state != "GESTURE")
            else:
                is_correct = (state == "GESTURE" and label == target)

            if is_correct:
                correct_frames.append((ts_ms, score))

        hold_count = len(hold_frames)
        fraction_correct = (len(correct_frames) / hold_count) if hold_count > 0 else 0.0

        if correct_frames:
            first_correct_ms = correct_frames[0][0] - hold_start_ts
            if target == "REST":
                max_score_correct = 1.0
            else:
                max_score_correct = max(s for _, s in correct_frames)
        else:
            first_correct_ms = None
            max_score_correct = 0.0

        results.append(
            TrialResult(
                trial=trial,
                target=target,
                hold_frame_count=hold_count,
                predicted_counts=dict(predicted_counts),
                fraction_correct=fraction_correct,
                first_correct_ms=first_correct_ms,
                max_score_correct=max_score_correct,
            )
        )

    return results


def summarize(rows, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """Computes summary metrics across targets for multiple confidence thresholds:

    - per target: trials count, mean frame recall, trial pass rate
      (pass = >=60% of HOLD frames equal the target with score>=thr;
       for REST, pass = zero GESTURE frames with score>=thr)
    - confusion matrix (target x predicted label) during HOLD
    - REST gesture-emission counts per threshold.
    """
    by_trial_hold = defaultdict(list)
    trial_target = {}
    all_targets = set()
    all_pred_labels = set()

    confusion_matrix = defaultdict(Counter)
    rest_emissions = {thr: 0 for thr in thresholds}

    for raw in rows:
        ts_ms, trial, target, phase, state, label, score = normalize_row(raw)
        trial_target[trial] = target
        all_targets.add(target)

        if phase == PHASE_HOLD:
            if state == "NO_HAND":
                pred = "NO_HAND"
            elif state == "NONE" or label == "None":
                pred = "None"
            else:
                pred = label

            all_pred_labels.add(pred)
            confusion_matrix[target][pred] += 1
            by_trial_hold[trial].append((state, label, score))

            if target == "REST" and state == "GESTURE":
                for thr in thresholds:
                    if score >= thr:
                        rest_emissions[thr] += 1

    # Target-level summary per threshold
    target_summaries = {}
    targets_ordered = [t for t in DEFAULT_TARGETS if t in all_targets]
    for t in sorted(all_targets):
        if t not in targets_ordered:
            targets_ordered.append(t)

    for target in targets_ordered:
        t_trials = [tr for tr, tgt in trial_target.items() if tgt == target]
        trial_count = len(t_trials)
        thr_metrics = {}

        for thr in thresholds:
            recalls = []
            passes = 0

            for tr in t_trials:
                frames = by_trial_hold.get(tr, [])
                total_f = len(frames)
                if total_f == 0:
                    recalls.append(0.0)
                    continue

                if target == "REST":
                    gesture_frames = sum(1 for st, lb, sc in frames if st == "GESTURE" and sc >= thr)
                    # Pass = zero GESTURE frames with score >= thr
                    is_pass = (gesture_frames == 0)
                    recall = (total_f - gesture_frames) / total_f
                else:
                    correct_frames = sum(
                        1 for st, lb, sc in frames if st == "GESTURE" and lb == target and sc >= thr
                    )
                    recall = correct_frames / total_f
                    is_pass = (recall >= 0.60)

                recalls.append(recall)
                if is_pass:
                    passes += 1

            mean_recall = sum(recalls) / len(recalls) if recalls else 0.0
            pass_rate = passes / trial_count if trial_count > 0 else 0.0
            thr_metrics[thr] = {
                "mean_recall": mean_recall,
                "pass_rate": pass_rate,
                "passed_trials": passes,
            }

        target_summaries[target] = {
            "trials": trial_count,
            "thresholds": thr_metrics,
        }

    return {
        "targets": target_summaries,
        "confusion_matrix": {t: dict(confusion_matrix[t]) for t in targets_ordered},
        "all_predicted_labels": sorted(all_pred_labels),
        "rest_emissions": rest_emissions,
    }


def debounce_sim(rows, hold_ms=400, thr=0.6):
    """Replays each trial to simulate debounce activation.

    A 'fire' = same GESTURE label with score >= thr continuously for >= hold_ms.
    Returns:
      target_fired (true positives on non-REST trials),
      wrong_gesture_fired (confusions on non-REST trials),
      rest_fires (false triggers during REST).
    """
    by_trial = defaultdict(list)
    trial_target = {}

    for raw in rows:
        ts_ms, trial, target, phase, state, label, score = normalize_row(raw)
        trial_target[trial] = target
        by_trial[trial].append((ts_ms, state, label, score))

    target_fired = 0
    wrong_gesture_fired = 0
    rest_fires = 0
    non_rest_trials = 0
    rest_trials = 0

    for trial, target in trial_target.items():
        frames = sorted(by_trial.get(trial, []), key=lambda x: x[0])
        if target == "REST":
            rest_trials += 1
        else:
            non_rest_trials += 1

        fired_gestures = set()
        candidate_label = None
        candidate_start_ts = None

        for ts_ms, state, label, score in frames:
            if state == "GESTURE" and score >= thr:
                if label == candidate_label:
                    if (ts_ms - candidate_start_ts) >= hold_ms:
                        fired_gestures.add(label)
                else:
                    candidate_label = label
                    candidate_start_ts = ts_ms
            else:
                candidate_label = None
                candidate_start_ts = None

        if target == "REST":
            if fired_gestures:
                rest_fires += len(fired_gestures)
        else:
            if target in fired_gestures:
                target_fired += 1
            confusions = [g for g in fired_gestures if g != target]
            if confusions:
                wrong_gesture_fired += len(confusions)

    return {
        "hold_ms": hold_ms,
        "thr": thr,
        "target_fired": target_fired,
        "non_rest_trials": non_rest_trials,
        "target_fired_pct": (target_fired / non_rest_trials) if non_rest_trials else 0.0,
        "wrong_gesture_fired": wrong_gesture_fired,
        "rest_fires": rest_fires,
        "rest_trials": rest_trials,
    }


def debounce_sim_multi(rows, thr=0.6, hold_ms_list=(250, 400, 600, 800)):
    """Runs debounce_sim across standard hold durations (250, 400, 600, 800 ms)."""
    return [debounce_sim(rows, hold_ms=h, thr=thr) for h in hold_ms_list]
