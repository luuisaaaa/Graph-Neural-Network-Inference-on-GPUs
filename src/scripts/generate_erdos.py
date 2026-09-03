#!/usr/bin/env python3
"""Generate a synthetic Erdos-Rényi graph and export it to CSR format."""

import argparse
import numpy as np
import networkx as nx
from pathlib import Path

from convert_dataset import build_csr
from converter import export_dataset, DEFAULT_OUTPUT_ROOT

def generate_erdos_renyi(num_nodes, p, feature_dim, num_classes=10):
    
    G = nx.fast_gnp_random_graph(num_nodes, p, directed=True)
    
    if G.number_of_edges() > 0:
        edges = np.array(G.edges()).T
    else:
        edges = np.empty((2, 0), dtype=np.int64)

    features = np.random.randn(num_nodes, feature_dim).astype(np.float32)

    labels = np.random.randint(0, num_classes, size=num_nodes).astype(np.int64)

    return edges, features, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, default=10000, help="Numero di nodi (default: 10000)")
    parser.add_argument("--p", type=float, default=0.001, help="Probabilità di creazione arco (default: 0.001)")
    parser.add_argument("--features", type=int, default=128, help="Dimensione del vettore di feature (default: 128)")
    parser.add_argument("--classes", type=int, default=10, help="Numero di classi per i nodi (default: 10)")
    parser.add_argument("--name", default="erdos_renyi_10k", help="Nome della cartella di output del dataset")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--undirected", action="store_true", help="Rende il grafo indiretto")

    args = parser.parse_args()

    edges, features, labels = generate_erdos_renyi(args.nodes, args.p, args.features, args.classes)

    rows, columns = build_csr(edges, args.nodes, args.undirected)

    export_dataset(args.name, rows, columns, features, labels, args.output_root)


if __name__ == "__main__":
    main()