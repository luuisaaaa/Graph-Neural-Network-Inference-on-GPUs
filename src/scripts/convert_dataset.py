#!/usr/bin/env python3
"""Convert Planetoid, OGB, or generic NPZ graphs to the project CSR format."""

import argparse
from pathlib import Path
import numpy as np
from converter import DEFAULT_OUTPUT_ROOT, export_dataset


def build_csr(edge_index, num_nodes, undirected=False):
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, num_edges]")
    source, destination = edges
    if undirected:
        source, destination = (np.concatenate((source, destination)),
                               np.concatenate((destination, source)))
    if np.any(source < 0) or np.any(source >= num_nodes) or np.any(destination < 0) \
            or np.any(destination >= num_nodes):
        raise ValueError("edge endpoint outside [0, num_nodes)")
    order = np.lexsort((destination, source))
    source, destination = source[order], destination[order]
    if source.size:
        keep = np.ones(source.size, dtype=bool)
        keep[1:] = (source[1:] != source[:-1]) | (destination[1:] != destination[:-1])
        source, destination = source[keep], destination[keep]
    counts = np.bincount(source, minlength=num_nodes)
    row_pointers = np.empty(num_nodes + 1, dtype=np.int64)
    row_pointers[0] = 0
    np.cumsum(counts, out=row_pointers[1:])
    return row_pointers, destination


def load_planetoid(name, raw_root):
    from torch_geometric.datasets import Planetoid
    data = Planetoid(root=str(raw_root), name=name)[0]
    return data.edge_index.cpu().numpy(), data.x.cpu().numpy(), data.y.cpu().numpy()


def load_ogb(name, raw_root):
    from ogb.nodeproppred import NodePropPredDataset
    graph, labels = NodePropPredDataset(name=name, root=str(raw_root))[0]
    return graph["edge_index"], graph["node_feat"], np.asarray(labels).reshape(-1)


def load_npz(path):
    archive = np.load(path)
    if "edge_index" in archive:
        edges = archive["edge_index"]
    elif "src" in archive and "dst" in archive:
        edges = np.vstack((archive["src"], archive["dst"]))
    else:
        raise ValueError("NPZ requires edge_index or src and dst arrays")
    feature_key = "node_features" if "node_features" in archive else "x"
    if feature_key not in archive:
        raise ValueError("NPZ requires node_features or x")
    label_key = "labels" if "labels" in archive else "y"
    labels = archive[label_key] if label_key in archive else None
    return edges, archive[feature_key], labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("planetoid", "ogb", "npz"))
    parser.add_argument("dataset", help="dataset name or path to an NPZ archive")
    parser.add_argument("--name", help="output dataset name")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--raw-root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "dataset" / "raw")
    parser.add_argument("--undirected", action="store_true",
                        help="insert reverse edges before CSR deduplication")
    args = parser.parse_args()

    if args.source == "planetoid":
        canonical = {"cora": "Cora", "citeseer": "CiteSeer", "pubmed": "PubMed"}
        key = args.dataset.lower()
        if key not in canonical:
            parser.error("Planetoid dataset must be Cora, CiteSeer, or PubMed")
        dataset_name = canonical[key]
        edges, features, labels = load_planetoid(dataset_name, args.raw_root)
    elif args.source == "ogb":
        dataset_name = args.dataset
        edges, features, labels = load_ogb(dataset_name, args.raw_root)
    else:
        source_path = Path(args.dataset)
        dataset_name = source_path.stem
        edges, features, labels = load_npz(source_path)

    features = np.asarray(features, dtype=np.float32)
    rows, columns = build_csr(edges, features.shape[0], args.undirected)
    export_dataset(args.name or dataset_name, rows, columns, features, labels, args.output_root)


if __name__ == "__main__":
    main()
