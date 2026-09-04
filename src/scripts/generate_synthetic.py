#!/usr/bin/env python3
"""Generate reproducible synthetic graphs and export them in project CSR format."""

import argparse
from pathlib import Path

import networkx as nx
import numpy as np

from convert_dataset import build_csr
from converter import DEFAULT_OUTPUT_ROOT, export_dataset


def graph_edges(graph):
    """Return NetworkX edges as a [2, E] int64 array, including the empty case."""
    edges = np.asarray(list(graph.edges()), dtype=np.int64)
    return edges.T if edges.size else np.empty((2, 0), dtype=np.int64)


def add_common_arguments(parser):
    parser.add_argument("--nodes", type=int, required=True, help="number of vertices")
    parser.add_argument("--features", type=int, default=128, help="features per vertex")
    parser.add_argument("--classes", type=int, default=10, help="number of synthetic labels")
    parser.add_argument("--seed", type=int, default=42, help="graph, feature, and label seed")
    parser.add_argument("--name", help="output directory name")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)


def validate_common(args):
    if args.nodes < 1:
        raise ValueError("--nodes must be positive")
    if args.features < 1:
        raise ValueError("--features must be positive")
    if args.classes < 1:
        raise ValueError("--classes must be positive")
    if args.classes > args.nodes:
        raise ValueError("--classes cannot exceed --nodes")


def generate_graph(args):
    if args.model == "erdos-renyi":
        if not 0.0 <= args.probability <= 1.0:
            raise ValueError("--probability must be in [0, 1]")
        graph = nx.fast_gnp_random_graph(
            args.nodes, args.probability, seed=args.seed, directed=args.directed)
        default_name = f"erdos_renyi_n{args.nodes}_p{args.probability:g}"
        # Undirected NetworkX edges are emitted once and need their reverse in CSR.
        make_undirected = not args.directed
    elif args.model == "barabasi-albert":
        if args.edges_per_new_node < 1 or args.edges_per_new_node >= args.nodes:
            raise ValueError("--edges-per-new-node must satisfy 1 <= m < nodes")
        graph = nx.barabasi_albert_graph(
            args.nodes, args.edges_per_new_node, seed=args.seed)
        default_name = f"barabasi_albert_n{args.nodes}_m{args.edges_per_new_node}"
        make_undirected = True
    else:
        if args.neighbors < 2 or args.neighbors >= args.nodes or args.neighbors % 2 != 0:
            raise ValueError("--neighbors must be even and satisfy 2 <= k < nodes")
        if not 0.0 <= args.rewiring_probability <= 1.0:
            raise ValueError("--rewiring-probability must be in [0, 1]")
        graph = nx.watts_strogatz_graph(
            args.nodes, args.neighbors, args.rewiring_probability, seed=args.seed)
        default_name = (
            f"watts_strogatz_n{args.nodes}_k{args.neighbors}_p"
            f"{args.rewiring_probability:g}"
        )
        make_undirected = True
    return graph, default_name, make_undirected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="model", required=True)

    erdos = subparsers.add_parser("erdos-renyi", help="Erdos-Renyi random graph")
    add_common_arguments(erdos)
    erdos.add_argument("--probability", "--p", type=float, default=0.001,
                       help="independent edge probability")
    direction = erdos.add_mutually_exclusive_group()
    direction.add_argument("--directed", action="store_true", help="generate a directed graph")
    direction.add_argument("--undirected", action="store_false", dest="directed",
                           help="generate an undirected graph (default)")
    erdos.set_defaults(directed=False)

    barabasi = subparsers.add_parser("barabasi-albert", help="scale-free graph")
    add_common_arguments(barabasi)
    barabasi.add_argument("--edges-per-new-node", "--m", type=int, default=4,
                          help="edges attached by every new vertex")

    watts = subparsers.add_parser("watts-strogatz", help="small-world graph")
    add_common_arguments(watts)
    watts.add_argument("--neighbors", "--k", type=int, default=8,
                       help="even number of initial ring neighbors")
    watts.add_argument("--rewiring-probability", "--p", type=float, default=0.1,
                       help="probability of rewiring each ring edge")

    args = parser.parse_args()
    try:
        validate_common(args)
        graph, default_name, make_undirected = generate_graph(args)
    except ValueError as error:
        parser.error(str(error))

    edges = graph_edges(graph)
    rows, columns = build_csr(edges, args.nodes, undirected=make_undirected)
    rng = np.random.default_rng(args.seed)
    features = rng.standard_normal((args.nodes, args.features)).astype(np.float32)
    labels = rng.integers(0, args.classes, size=args.nodes, dtype=np.int64)
    labels[:args.classes] = np.arange(args.classes, dtype=np.int64)
    rng.shuffle(labels)
    destination = export_dataset(
        args.name or default_name, rows, columns, features, labels, args.output_root)
    density = columns.size / (args.nodes * max(args.nodes - 1, 1))
    print(f"model={args.model} seed={args.seed} density={density:.8f}")
    print(f"directed_edges_in_csr={columns.size} output={destination}")


if __name__ == "__main__":
    main()
