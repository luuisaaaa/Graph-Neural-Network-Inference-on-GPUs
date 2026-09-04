#!/usr/bin/env python3
"""Generate fixed, reproducible GCN weights (no training is performed)."""

import argparse
import math
import random
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def read_metadata(dataset_dir):
    metadata = {}
    for line in (dataset_dir / "metadata.txt").read_text(encoding="ascii").splitlines():
        key, value = line.split(":", 1)
        metadata[key] = int(value)
    if "num_classes" not in metadata:
        labels = (int(value) for value in
                  (dataset_dir / "labels.txt").read_text(encoding="ascii").split())
        valid_labels = [label for label in labels if label >= 0]
        if not valid_labels:
            raise ValueError(f"cannot infer classes from {dataset_dir / 'labels.txt'}")
        metadata["num_classes"] = max(valid_labels) + 1
    return metadata


def model_dimensions(feature_dim, hidden_dim, num_classes, num_layers):
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if num_layers == 1:
        return [feature_dim, num_classes]
    return [feature_dim] + [hidden_dim] * (num_layers - 1) + [num_classes]


def generate(dataset, dataset_root, output_root, hidden_dim, num_layers, seed):
    dataset_dir = dataset_root / dataset
    metadata = read_metadata(dataset_dir)
    dimensions = model_dimensions(metadata["feature_dim"], hidden_dim,
                                  metadata["num_classes"], num_layers)
    destination = output_root / dataset / f"h{hidden_dim}_l{num_layers}_seed{seed}"
    destination.mkdir(parents=True, exist_ok=True)

    generator = random.Random(f"gcn-fixed-weights-v1:{dataset}:{seed}")
    manifest = ["format_version:1", f"dataset:{dataset}",
                f"seed:{seed}", f"num_layers:{num_layers}"]
    for layer, (in_dim, out_dim) in enumerate(zip(dimensions, dimensions[1:])):
        manifest.extend((f"layer_{layer}_in:{in_dim}",
                         f"layer_{layer}_out:{out_dim}"))
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        with (destination / f"layer_{layer}.txt").open("w", encoding="ascii") as output:
            for _ in range(in_dim * out_dim):
                output.write(f"{generator.uniform(-limit, limit):.9g}\n")

    (destination / "metadata.txt").write_text("\n".join(manifest) + "\n",
                                                encoding="ascii")
    print(f"{dataset}: {dimensions} -> {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="+", help="converted dataset directory names")
    parser.add_argument("--dataset-root", type=Path,
                        default=REPOSITORY_ROOT / "dataset" / "converted")
    parser.add_argument("--output-root", type=Path,
                        default=REPOSITORY_ROOT / "weights")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    for dataset in args.datasets:
        generate(dataset, args.dataset_root, args.output_root,
                 args.hidden_dim, args.num_layers, args.seed)


if __name__ == "__main__":
    main()
