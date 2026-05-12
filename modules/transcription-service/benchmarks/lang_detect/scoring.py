"""Metric computation: detector output vs ground truth.

Switch classification is the core idea here. Each detector switch is tagged
as one of:

  correct_initial      — first lock-on landed on the right language
  wrong_initial        — first lock-on landed on the wrong language
  correction_switch    — wrong → right (recovery — *helpful*)
  flap_switch          — right → wrong (breaking a good state — *harmful*)
  wrong_recovery       — wrong → still wrong (different wrong language)
  transition_caught    — switch matches a real ground-truth language change

This lets the ranking punish flaps (worst failure mode) without penalising
the detector for recovering from a bad initial guess — which the previous
version incorrectly conflated.
"""
from __future__ import annotations

from statistics import median

from .types import FixtureTrace, FrameTrace, GroundTruthSegment, RunResult


def ground_truth_at(t_ms: float, segments: list[GroundTruthSegment]) -> str | None:
    """Return the labelled language at time ``t_ms`` (or None if outside spans)."""
    for seg in segments:
        if seg.t_ms_start <= t_ms <= seg.t_ms_end:
            return seg.language
    return None


def real_transitions(segments: list[GroundTruthSegment]) -> list[tuple[float, str, str]]:
    """Return (t_ms, from_lang, to_lang) for each adjacent-segment language change."""
    sorted_segs = sorted(segments, key=lambda s: s.t_ms_start)
    out: list[tuple[float, str, str]] = []
    for prev, cur in zip(sorted_segs, sorted_segs[1:]):
        if prev.language != cur.language:
            out.append((cur.t_ms_start, prev.language, cur.language))
    return out


def _is_correct(lang: str | None, truth: str | None) -> bool:
    """A language is 'correct' if it matches truth at that t. None truth → vacuous."""
    return lang is not None and truth is not None and lang == truth


def score_run(
    fixture: FixtureTrace,
    detector_states: list[tuple[FrameTrace, str | None, bool]],
    params,
) -> RunResult:
    """Compute classified metrics from a per-frame detector trace.

    ``detector_states[i] = (frame, current_language_after_frame, switched_this_frame)``.
    """
    gt = fixture.ground_truth
    transitions = real_transitions(gt)

    # Pass 1: timing + frame-impact metrics.
    time_to_correct: float | None = None
    final_language: str | None = None
    frames_wrong = 0
    for frame, lang_after, _switched in detector_states:
        truth = ground_truth_at(frame.t_ms, gt)
        if _is_correct(lang_after, truth) and time_to_correct is None:
            time_to_correct = frame.t_ms
        if truth is not None and lang_after is not None and lang_after != truth:
            frames_wrong += 1
        final_language = lang_after

    if detector_states:
        last_frame, last_lang, _ = detector_states[-1]
        correct_at_end = _is_correct(last_lang, ground_truth_at(last_frame.t_ms, gt))
    else:
        correct_at_end = False

    # Pass 2: classify each switch with the language *before* and *after*.
    correct_initial = False
    wrong_initial = False
    correction = 0
    flap = 0
    wrong_recovery = 0
    transitions_caught: set[int] = set()
    caught_latencies: list[float] = []

    tolerance_ms = params.min_dwell_ms + 5_000
    prev_lang: str | None = None

    for frame, lang_after, switched in detector_states:
        if not switched:
            prev_lang = lang_after
            continue

        truth_after = ground_truth_at(frame.t_ms, gt)
        # The "before" truth — use truth at this t too, since switches happen
        # at a discrete frame; pre-state was held until this moment.
        was_correct = _is_correct(prev_lang, truth_after)
        now_correct = _is_correct(lang_after, truth_after)

        # 1) Is this an initial lock-on?
        if prev_lang is None:
            if now_correct:
                correct_initial = True
            else:
                wrong_initial = True
            # Fall through — initial lock can also coincide with a real
            # transition if t_ms ≈ a transition start. Check below.

        # 2) Does this switch match a real transition (within tolerance,
        #    correct target)? Independent of correct/flap classification.
        for i, (tr_t, _tr_from, tr_to) in enumerate(transitions):
            if i in transitions_caught:
                continue
            if lang_after == tr_to and abs(frame.t_ms - tr_t) <= tolerance_ms and frame.t_ms >= tr_t:
                transitions_caught.add(i)
                caught_latencies.append(frame.t_ms - tr_t)
                break

        # 3) Correction / flap / wrong-recovery (only for non-initial switches).
        if prev_lang is not None:
            if not was_correct and now_correct:
                correction += 1
            elif was_correct and not now_correct:
                flap += 1
            elif not was_correct and not now_correct:
                wrong_recovery += 1
            # was_correct and now_correct = no-op switch (shouldn't really
            # happen — switching to the same language doesn't count).

        prev_lang = lang_after

    missed = len(transitions) - len(transitions_caught)
    latency_median = median(caught_latencies) if caught_latencies else None

    return RunResult(
        fixture_id=fixture.fixture_id,
        params=params,
        correct_at_end=correct_at_end,
        time_to_correct_lang_ms=time_to_correct,
        final_language=final_language,
        correct_initial=correct_initial,
        wrong_initial=wrong_initial,
        correction_switches=correction,
        flap_switches=flap,
        wrong_recovery_switches=wrong_recovery,
        transitions_caught=len(transitions_caught),
        missed_transitions=missed,
        switch_latency_ms_median=latency_median,
        frames_total=len(detector_states),
        frames_with_wrong_lang=frames_wrong,
    )
