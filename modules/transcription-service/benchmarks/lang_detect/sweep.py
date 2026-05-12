"""Stage 3 — grid sweep over DetectorParams across fixtures.

Reads a sweep config YAML, materialises the Cartesian product of all
listed parameter values, replays each (fixture × params) combination,
writes a ranked TSV and an append-only JSONL index — matching the
conventions of benchmarks/pipeline_benchmark.py.

Sort order (ascending priority): ``--sort-by`` columns left-to-right,
with the rest of RunResult breaking ties alphabetically.

CLI:
    uv run python -m benchmarks.lang_detect.sweep \\
        --config benchmarks/lang_detect/sweep_config.yaml \\
        --output-dir benchmarks/lang_detect/results
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import asdict
from pathlib import Path

import yaml  # type: ignore

from .replay import load_ground_truth, load_trace, run_replay
from .types import DetectorParams, FixtureTrace, RunResult


def _expand_grid(grid: dict) -> list[DetectorParams]:
    """Cartesian product of all list-valued keys in ``grid``."""
    keys = sorted(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    out: list[DetectorParams] = []
    for combo in itertools.product(*value_lists):
        kwargs = dict(zip(keys, combo))
        out.append(DetectorParams(**kwargs))
    return out


def _flatten_result(r: RunResult) -> dict:
    """Flatten RunResult + params into a single row for TSV/JSONL output."""
    row: dict = asdict(r)
    params = row.pop("params")
    for k, v in params.items():
        row[f"param_{k}"] = v
    return row


def _sort_key(row: dict, sort_by: list[str]) -> tuple:
    """Tuple key for sorting rows. None → infinity so 'never' ranks worst."""
    out = []
    for col in sort_by:
        v = row.get(col)
        if v is None:
            out.append(float("inf"))
        elif isinstance(v, bool):
            # ``correct_at_end=True`` should rank higher than False.
            out.append(0 if v else 1)
        else:
            out.append(v)
    return tuple(out)


def run_sweep(config: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    grid = config.get("param_grid", {})
    combos = _expand_grid(grid)
    fixture_specs = config.get("fixtures", [])
    gt_path = Path(config["ground_truth"])
    sort_by = config.get(
        "sort_by",
        ["flap_switches", "wrong_initial", "frames_with_wrong_lang", "time_to_correct_lang_ms"],
    )

    rows: list[dict] = []
    print(f"Sweep: {len(combos)} param combos × {len(fixture_specs)} fixtures "
          f"= {len(combos) * len(fixture_specs)} runs")

    for spec in fixture_specs:
        trace_path = Path(spec["trace"])
        fixture_id = spec.get("id") or trace_path.stem
        frames = load_trace(trace_path)
        gt = load_ground_truth(gt_path, fixture_id)
        fixture = FixtureTrace(
            fixture_id=fixture_id,
            wav_path=trace_path,
            frames=frames,
            ground_truth=gt,
            total_duration_ms=frames[-1].t_ms if frames else 0.0,
        )

        for params in combos:
            result = run_replay(fixture, params)
            rows.append(_flatten_result(result))

    rows.sort(key=lambda r: _sort_key(r, sort_by))

    # TSV
    tsv_path = output_dir / f"sweep_{ts}.tsv"
    if rows:
        headers = list(rows[0].keys())
        with tsv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=headers, delimiter="\t")
            w.writeheader()
            for row in rows:
                w.writerow({k: ("" if v is None else v) for k, v in row.items()})

    # JSONL index (append-only, for cross-run regression tracking).
    jsonl_path = output_dir / "sweep_index.jsonl"
    with jsonl_path.open("a") as f:
        for row in rows:
            row["_run_ts"] = ts
            f.write(json.dumps(row, default=str) + "\n")

    print(f"\nTop 10 rows (sorted by {sort_by}):")
    for row in rows[:10]:
        print(f"  fixture={row['fixture_id']:<22} "
              f"margin={row['param_confidence_margin']:<4} "
              f"frames={row['param_min_dwell_frames']:<3} "
              f"dwell_ms={row['param_min_dwell_ms']:<7} "
              f"tiebreak={row['param_script_tiebreaker_enabled']:<6} "
              f"init_gate={row['param_initial_confidence_threshold']:<4} "
              f"→ correct_end={row['correct_at_end']} "
              f"ttc={row['time_to_correct_lang_ms']} "
              f"flap={row['flap_switches']} "
              f"correction={row['correction_switches']} "
              f"wrong_init={row['wrong_initial']} "
              f"wrong_frames={row['frames_with_wrong_lang']}")
    print(f"\nFull results: {tsv_path}")
    return tsv_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path,
                    default=Path("benchmarks/lang_detect/results"))
    args = ap.parse_args()

    with args.config.open() as f:
        config = yaml.safe_load(f)
    run_sweep(config, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
