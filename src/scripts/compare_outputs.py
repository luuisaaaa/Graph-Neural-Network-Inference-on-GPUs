#!/usr/bin/env python3
"""Compare probability files produced through GCN_OUTPUT_FILE."""

import argparse
from pathlib import Path
import numpy as np


def load_output(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8") as stream:
        header = stream.readline().split()
        if len(header) != 2:
            raise ValueError(f"Invalid header in {path}")
        nodes, classes = map(int, header)
        values = np.loadtxt(stream, dtype=np.float64)
    values = np.asarray(values).reshape(nodes, classes)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()

    reference = load_output(args.reference)
    candidate = load_output(args.candidate)
    if reference.shape != candidate.shape:
        raise SystemExit(f"FAIL shape: {reference.shape} != {candidate.shape}")

    absolute = np.abs(reference - candidate)
    predictions_reference = np.argmax(reference, axis=1)
    predictions_candidate = np.argmax(candidate, axis=1)
    mismatches = int(np.count_nonzero(predictions_reference != predictions_candidate))
    passed = np.allclose(reference, candidate, atol=args.atol, rtol=args.rtol)
    print(f"{'PASS' if passed else 'FAIL'} max_abs_error={absolute.max():.9g} "
          f"mean_abs_error={absolute.mean():.9g} prediction_mismatches={mismatches}")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
