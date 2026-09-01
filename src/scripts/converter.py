"""Validation and export helpers for the C++ CSR dataset format."""

from pathlib import Path
from typing import Optional
import numpy as np

DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "converted"


def export_dataset(dataset_name, row_pointers, column_indices, node_features,
                   labels=None, output_root: Optional[Path] = None):
    row_pointers = np.asarray(row_pointers, dtype=np.int64).reshape(-1)
    column_indices = np.asarray(column_indices, dtype=np.int64).reshape(-1)
    node_features = np.asarray(node_features, dtype=np.float32)
    if node_features.ndim != 2:
        raise ValueError("node_features must have shape [num_nodes, feature_dim]")
    num_nodes, feature_dim = node_features.shape
    num_edges = column_indices.size
    if row_pointers.size != num_nodes + 1:
        raise ValueError("row_pointers must contain num_nodes + 1 entries")
    if row_pointers[0] != 0 or row_pointers[-1] != num_edges:
        raise ValueError("CSR pointers must start at 0 and end at num_edges")
    if np.any(row_pointers[1:] < row_pointers[:-1]):
        raise ValueError("row_pointers must be non-decreasing")
    if np.any(column_indices < 0) or np.any(column_indices >= num_nodes):
        raise ValueError("column index outside [0, num_nodes)")

    if labels is None:
        labels = np.full(num_nodes, -1, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if labels.size != num_nodes:
        raise ValueError("labels must contain one value per node")
    valid_labels = labels[labels >= 0]
    num_classes = int(valid_labels.max() + 1) if valid_labels.size else 0

    root = Path(output_root) if output_root is not None else DEFAULT_OUTPUT_ROOT
    destination = root / dataset_name
    destination.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination / "row_pointers.txt", row_pointers, fmt="%d")
    np.savetxt(destination / "column_indices.txt", column_indices, fmt="%d")
    np.savetxt(destination / "node_features.txt", node_features.reshape(-1), fmt="%.9g")
    np.savetxt(destination / "labels.txt", labels, fmt="%d")
    (destination / "metadata.txt").write_text(
        f"num_nodes:{num_nodes}\nnum_edges:{num_edges}\n"
        f"feature_dim:{feature_dim}\nnum_classes:{num_classes}\n", encoding="ascii")
    print(f"Dataset '{dataset_name}' exported to {destination}")
    print(f"nodes={num_nodes} edges={num_edges} features={feature_dim} classes={num_classes}")
    return destination
