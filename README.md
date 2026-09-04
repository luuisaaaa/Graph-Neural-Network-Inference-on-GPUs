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
nodes/s, messages/s, checksums, and accuracy. Data loading and host-device
copies are excluded from `inference_ms`; CUDA uses events so asynchronous
kernel execution is measured correctly.

Set `GCN_OUTPUT_FILE` to save every final probability:

    GCN_OUTPUT_FILE=/tmp/sequential.txt ./sequential Cora 16 7 2
    GCN_OUTPUT_FILE=/tmp/candidate.txt ./candidate Cora 16 7 2
    python src/scripts/compare_outputs.py /tmp/sequential.txt /tmp/candidate.txt

The comparator uses numerical tolerances because parallel floating-point
reductions may sum messages in a different order.

Fixed pre-loaded weights
------------------------

## Fixed pre-loaded weights

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
