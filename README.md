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

    GCN_OUTPUT_FILE=/tmp/sequential.txt ./sequential Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
    GCN_OUTPUT_FILE=/tmp/candidate.txt ./candidate Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
    python src/scripts/compare_outputs.py /tmp/sequential.txt /tmp/candidate.txt

The comparator uses numerical tolerances because parallel floating-point
reductions may sum messages in a different order.

Memory fields distinguish CSR topology, input features, labels, weights,
temporary working buffers, and allocations on the CUDA device. The total is an
algorithmic estimate based on owned arrays; allocator overhead and process/runtime
memory are intentionally excluded.

Repeated benchmarks
-------------------

`benchmark.h` measures one inference run. The external runner performs warm-up
runs, repeats every configured command, verifies stable checksums, and writes raw
and statistical CSV files:

    python src/scripts/run_benchmarks.py --config benchmark_config.example.json

The default protocol is 3 warm-up runs followed by 10 measured runs. Edit or
copy the JSON file to add CPU, CUDA, dataset, feature-size, depth, and OpenMP
thread-count configurations. Results are written under `results/` and include
the Git commit, host, and platform so measurements remain traceable.

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

Fixed pre-loaded weights
------------------------

Inference does not train the GCN. Generate the fixed model files once after
converting the datasets:

```bash
python src/scripts/generate_weights.py Cora PubMed ogbn-arxiv \
  --hidden-dim 16 --num-layers 2 --seed 42
```

The command creates one validated model per dataset under `weights/`. The same
files must be passed to every sequential, CPU, and CUDA implementation. Weight
loading happens before the measured inference region.

Example for the sequential executable, run from `src/GCN/sequential`:

```bash
g++ -O3 -std=c++17 main.cpp ../utilities/graph.cpp ../utilities/inference.cpp -o sequential
./sequential Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
./sequential PubMed 16 3 2 ../../../weights/PubMed/h16_l2_seed42
./sequential ogbn-arxiv 16 40 2 ../../../weights/ogbn-arxiv/h16_l2_seed42
```
