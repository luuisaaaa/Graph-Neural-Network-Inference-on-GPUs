#!/usr/bin/env python3
"""Run RESULT-producing GCN executables repeatedly and write CSV reports."""

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PERFORMANCE_FIELDS = (
    "inference_ms",
    "graph_nodes_per_second",
    "node_updates_per_second",
    "messages_per_second",
)
MEMORY_FIELDS = (
    "topology_memory_bytes",
    "feature_memory_bytes",
    "label_memory_bytes",
    "weight_memory_bytes",
    "working_memory_bytes",
    "device_memory_bytes",
    "estimated_total_memory_bytes",
)
CHECKSUM_FIELDS = ("probability_checksum", "prediction_checksum")
REQUIRED_FIELDS = ("implementation",) + PERFORMANCE_FIELDS + MEMORY_FIELDS + CHECKSUM_FIELDS


def parse_result(stdout):
    lines = [line.strip() for line in stdout.splitlines()
             if line.strip().startswith("RESULT ")]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one RESULT line, found {len(lines)}")
    result = {}
    for token in shlex.split(lines[0][len("RESULT "):]):
        if "=" not in token:
            raise ValueError(f"invalid RESULT token: {token!r}")
        key, value = token.split("=", 1)
        result[key] = value
    missing = [field for field in REQUIRED_FIELDS if field not in result]
    if missing:
        raise ValueError("missing RESULT fields: " + ", ".join(missing))
    for field in PERFORMANCE_FIELDS + MEMORY_FIELDS + CHECKSUM_FIELDS:
        try:
            numeric = float(result[field])
        except ValueError as error:
            raise ValueError(f"RESULT field {field} is not numeric") from error
        if not math.isfinite(numeric):
            raise ValueError(f"RESULT field {field} is not finite")
    return result


def run_once(command, cwd, environment, timeout):
    completed = subprocess.run(
        command, cwd=cwd, env=environment, capture_output=True, text=True,
        timeout=timeout, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")
    try:
        return parse_result(completed.stdout)
    except ValueError as error:
        raise RuntimeError(
            f"cannot parse benchmark output: {error}\nstdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}") from error


def resolve_case(case, default_warmups, default_runs):
    for field in ("name", "dataset", "command"):
        if field not in case:
            raise ValueError(f"benchmark case is missing {field!r}")
    if not isinstance(case["command"], list) or not case["command"]:
        raise ValueError(f"case {case['name']!r}: command must be a non-empty list")
    warmups = int(case.get("warmups", default_warmups))
    runs = int(case.get("runs", default_runs))
    if warmups < 0 or runs < 2:
        raise ValueError(f"case {case['name']!r}: warmups >= 0 and runs >= 2 required")
    threads = case.get("threads", [None])
    if not threads:
        threads = [None]
    for value in threads:
        if value is not None and int(value) < 1:
            raise ValueError(f"case {case['name']!r}: thread counts must be positive")
    return warmups, runs, threads


def check_stability(rows, relative_tolerance):
    prediction = rows[0]["prediction_checksum"]
    if any(row["prediction_checksum"] != prediction for row in rows[1:]):
        raise RuntimeError("prediction_checksum changed between measured runs")
    reference = float(rows[0]["probability_checksum"])
    for row in rows[1:]:
        if not math.isclose(float(row["probability_checksum"]), reference,
                            rel_tol=relative_tolerance, abs_tol=1e-8):
            raise RuntimeError("probability_checksum changed beyond the configured tolerance")
    for field in MEMORY_FIELDS:
        if any(row[field] != rows[0][field] for row in rows[1:]):
            raise RuntimeError(f"{field} changed between measured runs")


def git_commit():
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT,
        capture_output=True, text=True, check=False)
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=600.0,
                        help="timeout in seconds for each process")
    parser.add_argument("--checksum-rtol", type=float, default=1e-6)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    cases = [case for case in config.get("benchmarks", []) if case.get("enabled", True)]
    if not cases:
        parser.error("the configuration contains no enabled benchmarks")

    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    common = {
        "timestamp_utc": timestamp,
        "git_commit": git_commit(),
        "host": platform.node(),
        "platform": platform.platform(),
    }
    raw_rows = []
    summary_rows = []

    for case in cases:
        warmups, runs, thread_counts = resolve_case(case, args.warmups, args.runs)
        cwd = REPOSITORY_ROOT / case.get("cwd", ".")
        command = [str(part) for part in case["command"]]
        if not cwd.is_dir():
            raise RuntimeError(f"case {case['name']!r}: cwd does not exist: {cwd}")

        for thread_count in thread_counts:
            environment = os.environ.copy()
            environment.update({str(k): str(v) for k, v in case.get("env", {}).items()})
            if thread_count is not None:
                environment["OMP_NUM_THREADS"] = str(thread_count)
            environment.pop("GCN_OUTPUT_FILE", None)
            label = f"{case['name']} threads={thread_count or 'n/a'}"
            print(f"[benchmark] {label}: {warmups} warm-up, {runs} measured")

            for warmup in range(warmups):
                run_once(command, cwd, environment, args.timeout)
                print(f"  warm-up {warmup + 1}/{warmups}", end="\r", flush=True)

            measured = []
            for run_index in range(1, runs + 1):
                result = run_once(command, cwd, environment, args.timeout)
                row = dict(common)
                row.update({
                    "case": case["name"],
                    "dataset": case["dataset"],
                    "hidden_dim": case.get("hidden_dim", ""),
                    "num_classes": case.get("num_classes", ""),
                    "num_layers": case.get("num_layers", ""),
                    "threads": thread_count or "",
                    "run": run_index,
                    "command": shlex.join(command),
                })
                row.update(result)
                measured.append(result)
                raw_rows.append(row)
                print(f"  run {run_index}/{runs}: {result['inference_ms']} ms")

            check_stability(measured, args.checksum_rtol)
            summary = dict(common)
            summary.update({
                "case": case["name"],
                "implementation": measured[0]["implementation"],
                "dataset": case["dataset"],
                "hidden_dim": case.get("hidden_dim", ""),
                "num_classes": case.get("num_classes", ""),
                "num_layers": case.get("num_layers", ""),
                "threads": thread_count or "",
                "runs": runs,
            })
            for field in PERFORMANCE_FIELDS:
                values = [float(row[field]) for row in measured]
                summary[f"{field}_min"] = min(values)
                summary[f"{field}_mean"] = statistics.fmean(values)
                summary[f"{field}_median"] = statistics.median(values)
                summary[f"{field}_stdev"] = statistics.stdev(values)
                summary[f"{field}_max"] = max(values)
            for field in MEMORY_FIELDS + CHECKSUM_FIELDS:
                summary[field] = measured[0][field]
            summary_rows.append(summary)

    raw_fields = (
        "timestamp_utc", "git_commit", "host", "platform", "case", "implementation",
        "dataset", "hidden_dim", "num_classes", "num_layers", "threads", "run", "command",
    ) + PERFORMANCE_FIELDS + MEMORY_FIELDS + CHECKSUM_FIELDS
    summary_fields = (
        "timestamp_utc", "git_commit", "host", "platform", "case", "implementation",
        "dataset", "hidden_dim", "num_classes", "num_layers", "threads", "runs",
    ) + tuple(f"{field}_{stat}" for field in PERFORMANCE_FIELDS
              for stat in ("min", "mean", "median", "stdev", "max")) + \
        MEMORY_FIELDS + CHECKSUM_FIELDS

    raw_path = args.output_dir / "benchmark_raw.csv"
    summary_path = args.output_dir / "benchmark_summary.csv"
    write_csv(raw_path, raw_fields, raw_rows)
    write_csv(summary_path, summary_fields, summary_rows)
    print(f"Raw measurements: {raw_path}")
    print(f"Statistical summary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
