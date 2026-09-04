Graph Neural Network Inference on GPUs
======================================

Dataset conversion
------------------

All converters produce `dataset/converted/<name>/` with CSR topology, dense
features, labels, and metadata. Examples:

    python src/scripts/convert_dataset.py planetoid Cora
    python src/scripts/convert_dataset.py planetoid PubMed
    python src/scripts/convert_dataset.py ogb ogbn-arxiv
    python src/scripts/convert_dataset.py npz graph.npz --name MyGraph --undirected

NPZ inputs must contain `edge_index` (or `src` and `dst`), `node_features`
(or `x`), and optionally `labels` (or `y`).

Comparable outputs and benchmarks
---------------------------------

Executables print a machine-readable `RESULT` line containing inference time,
graph nodes/s, node updates/s, messages/s, memory estimates, and checksums. Data loading and host-device
copies are excluded from `inference_ms`; CUDA uses events so asynchronous
kernel execution is measured correctly.

Set `GCN_OUTPUT_FILE` to save every final probability:

    GCN_OUTPUT_FILE=/tmp/sequential.txt ./sequential Cora 16 7 2
    GCN_OUTPUT_FILE=/tmp/candidate.txt ./candidate Cora 16 7 2
    python src/scripts/compare_outputs.py /tmp/sequential.txt /tmp/candidate.txt

The comparator uses numerical tolerances because parallel floating-point
reductions may sum messages in a different order.

Memory fields distinguish CSR topology, input features, labels, weights,
temporary working buffers, and allocations on the CUDA device. The total is an
algorithmic estimate based on owned arrays; allocator overhead and process/runtime
memory are intentionally excluded.

Synthetic graphs
----------------

The unified generator creates reproducible Erdos-Renyi, Barabasi-Albert
(scale-free), and Watts-Strogatz (small-world) datasets:

    python src/scripts/generate_synthetic.py erdos-renyi --nodes 10000 --p 0.001 --features 128
    python src/scripts/generate_synthetic.py barabasi-albert --nodes 10000 --m 4 --features 128
    python src/scripts/generate_synthetic.py watts-strogatz --nodes 10000 --k 8 --p 0.1 --features 128

Use `--seed` to reproduce topology, features, and labels exactly. Undirected
models are exported with both directions in CSR so all inference strategies
observe the same neighborhood relation.
