#!/usr/bin/env python3
"""Backward-compatible Planetoid converter; prefer convert_dataset.py."""

import sys
from pathlib import Path
from convert_dataset import build_csr, load_planetoid
from converter import DEFAULT_OUTPUT_ROOT, export_dataset


def main():
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python load_cora.py [Cora|CiteSeer|PubMed]")
    canonical = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
    name = canonical.get(sys.argv[1].lower())
    if name is None:
        raise SystemExit("Dataset supportati: Cora, CiteSeer, PubMed")
    raw_root = Path(__file__).resolve().parents[2] / "dataset" / "raw"
    edges, features, labels = load_planetoid(name, raw_root)
    rows, columns = build_csr(edges, features.shape[0])
    export_dataset(name, rows, columns, features, labels, DEFAULT_OUTPUT_ROOT)


if __name__ == "__main__":
    main()
