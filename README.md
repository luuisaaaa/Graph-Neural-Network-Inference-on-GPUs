Graph Neural Network Inference on GPUs
======================================

This project contains a C++/CUDA inference engine for a GCN-style Graph
Neural Network. It focuses on full-batch inference on one static graph:
no training is performed and all layer weights are loaded from text files before
the measured inference region.

Implemented variants:

- sequential sparse CSR baseline;
- sequential dense adjacency baseline;
- CPU vertex-parallel implementation;
- CPU edge-parallel implementation;
- CPU message-batching implementation;
- CPU dense-parallel implementation;
- CUDA vertex-parallel basic implementation;
- CUDA vertex-parallel improved implementation;
- CUDA vertex-parallel FP16 compressed implementation;
- CUDA edge-parallel basic implementation;
- CUDA edge-parallel improved implementation;
- CUDA message-batching basic implementation;
- CUDA message-batching improved implementation.


Project layout
--------------

```text
dataset/converted/                         Converted datasets in CSR format
weights/                                   Fixed pre-loaded GCN weights
src/GCN/utilities/                         Shared graph, weight, and metric utilities
src/GCN/sequential/                        Sequential CSR implementation
src/GCN/sequential_dense/                  Sequential dense adjacency implementation
src/GCN/CPU_parallelization/               CPU parallel implementations
src/GCN/CUDA/                              CUDA implementations
README.md                                  Compilation and execution instructions
```


Requirements
------------

CPU implementations require:

- a C++17 compiler;
- OpenMP support for the CPU vertex, edge, and message-batching variants;
- POSIX threads support for the CPU dense-parallel variant.

CUDA implementations require:

- NVIDIA CUDA Toolkit;
- `nvcc`;
- a CUDA-capable NVIDIA GPU.



Dataset format
--------------

All implementations expect a converted dataset directory with this structure:

```text
dataset/converted/<dataset_name>/
  metadata.txt
  row_pointers.txt
  column_indices.txt
  node_features.txt
  labels.txt
```

The files are plain text.

`metadata.txt` contains one `key:value` entry per line. Required keys:

```text
num_nodes:<number_of_nodes>
num_edges:<number_of_edges>
feature_dim:<input_feature_dimension>
num_classes:<number_of_classes>
```

`row_pointers.txt` contains the CSR row pointer array, one integer per line. Its
length must be `num_nodes + 1`. The first value must be 0 and the last value
must be `num_edges`.

`column_indices.txt` contains the CSR column index array, one integer per line.
Its length must be `num_edges`.

`node_features.txt` contains the dense node-feature matrix flattened in
row-major order, one float per line. Its length must be:

```text
num_nodes * feature_dim
```

`labels.txt` contains one integer label per node. Its length must be
`num_nodes`. Labels are used only for output/correctness checks, not for
training.

The sparse implementations use the CSR representation directly. Dense
implementations load the same dataset and materialize an `N x N` adjacency
matrix internally; therefore, dense runs are intended only for small graphs.


Weights format
--------------

All implementations load fixed weights from:

```text
weights/<dataset_name>/h<hidden_dim>_l<num_layers>_seed<seed>/
  metadata.txt
  layer_0.txt
  layer_1.txt
  ...
```


Each `layer_i.txt` file stores the layer weight matrix as plain text, one float
per line, in row-major order. The same weight directory must be passed to all
implementations being compared. The loader checks that the dataset name, number
of layers, and layer dimensions match the command-line parameters.


Execution parameters
--------------------

Most executables use this interface:

```text
./program <dataset_name> <hidden_dim> <num_classes> <num_layers> <weights_dir>
```

Parameters:

- `dataset_name`: name of the directory under `dataset/converted/`;
- `hidden_dim`: hidden feature dimension used by intermediate GCN layers;
- `num_classes`: output classes;
- `num_layers`: number of GCN layers;
- `weights_dir`: directory containing `metadata.txt` and `layer_i.txt` files.

Additional optional parameters:

- CPU/CUDA message-batching implementations:
  `./program ... [message_batch_size]`, default `4096`;
- sequential dense implementation:
  `./sequential_dense ... [max_dense_memory_mb]`, default `1024`;
- CPU dense-parallel implementation:
  `./dense_cpu ... [num_threads] [max_dense_memory_mb]`.

For OpenMP CPU implementations, set the thread count with `OMP_NUM_THREADS`.


Compile and run: sequential CPU
-------------------------------

Run from `src/GCN/sequential`:

```bash
cd src/GCN/sequential
g++ -O3 -std=c++17 main.cpp ../utilities/graph.cpp ../utilities/inference.cpp -o sequential
./sequential Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
```


Compile and run: sequential dense CPU
-------------------------------------

Run from `src/GCN/sequential_dense`:

```bash
cd src/GCN/sequential_dense
g++ -O3 -std=c++17 main.cpp ../utilities/graph.cpp ../utilities/inference.cpp -o sequential_dense
./sequential_dense Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42 1024
```

The final argument is the maximum dense adjacency memory allowed, in MB.


Compile and run: CPU parallel implementations
---------------------------------------------

Vertex-parallel, from `src/GCN/CPU_parallelization/vertex_parallelization`:

```bash
cd src/GCN/CPU_parallelization/vertex_parallelization
g++ -O3 -std=c++17 -fopenmp main.cpp ../../utilities/graph.cpp ../../utilities/inference.cpp -o vertex_cpu
OMP_NUM_THREADS=8 ./vertex_cpu Cora 16 7 2 ../../../../weights/Cora/h16_l2_seed42
```

Edge-parallel, from `src/GCN/CPU_parallelization/edge_parallelization`:

```bash
cd src/GCN/CPU_parallelization/edge_parallelization
g++ -O3 -std=c++17 -fopenmp main.cpp ../../utilities/graph.cpp ../../utilities/inference.cpp -o program
OMP_NUM_THREADS=8 ./program Cora 16 7 2 ../../../../weights/Cora/h16_l2_seed42
```

Message-batching, from `src/GCN/CPU_parallelization/message_batching`:

```bash
cd src/GCN/CPU_parallelization/message_batching
g++ -O3 -std=c++17 -fopenmp main.cpp ../../utilities/graph.cpp ../../utilities/inference.cpp -o message_cpu
OMP_NUM_THREADS=8 ./message_cpu Cora 16 7 2 ../../../../weights/Cora/h16_l2_seed42 4096
```

Dense-parallel, from `src/GCN/CPU_parallelization/dense_parallelization`:

```bash
cd src/GCN/CPU_parallelization/dense_parallelization
g++ -O3 -std=c++17 -pthread main.cpp ../../utilities/graph.cpp ../../utilities/inference.cpp -o dense_cpu
./dense_cpu Cora 16 7 2 ../../../../weights/Cora/h16_l2_seed42 8 1024
```

In the dense-parallel command, `8` is the requested number of CPU threads and
`1024` is the dense adjacency memory limit in MB.


Compile and run: CUDA implementations
-------------------------------------

Each CUDA implementation is compiled from its own directory with:

```bash
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
```

Vertex basic:

```bash
cd src/GCN/CUDA/vertex_parallelization/basic_version
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42
```

Vertex improved:

```bash
cd src/GCN/CUDA/vertex_parallelization/improved_version
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42
```

Vertex compressed FP16:

```bash
cd src/GCN/CUDA/vertex_parallelization/compressed_fp16
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42
```

Edge basic:

```bash
cd src/GCN/CUDA/edge_parallelization/basic_version
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42
```

Edge improved:

```bash
cd src/GCN/CUDA/edge_parallelization/improved_version
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42
```

Message-batching basic:

```bash
cd src/GCN/CUDA/message_batching/basic_version
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42 4096
```

Message-batching improved:

```bash
cd src/GCN/CUDA/message_batching/improved_version
nvcc -O3 -std=c++17 main.cu ../../../utilities/graph.cpp ../../../utilities/inference.cpp -o program
./program Cora 16 7 2 ../../../../../weights/Cora/h16_l2_seed42 4096
```


Benchmark output
----------------

Each executable prints a `RESULT` block with:

- `implementation`;
- `inference_ms`;
- `graph_nodes_per_second`;
- `node_updates_per_second`;
- `messages_per_second`;
- memory estimates for topology, features, labels, weights, working buffers, and
  CUDA device allocations;
- `estimated_total_memory_bytes`;
- `probability_checksum`;
- `prediction_checksum`.

The reported time measures only GCN forward inference. Dataset loading, weight
loading, preprocessing, initial CUDA H2D transfers, and final CUDA D2H transfers
are outside the measured inference interval. This convention is used so that CPU
and CUDA implementations are compared on the computation performed by the GCN
forward pass, not on setup or I/O costs.


Correctness checks
------------------

Equal `prediction_checksum` values indicate that the final predicted
classes are the same. Small changes in `probability_checksum` may occur because
parallel floating-point reductions can sum values in a different order.

For a node-by-node numerical comparison, set `GCN_OUTPUT_FILE` before running an
executable. The program will write the full probability matrix to the selected
path:

```bash
cd src/GCN/sequential
GCN_OUTPUT_FILE=/tmp/gcn_sequential.txt ./sequential Cora 16 7 2 ../../../weights/Cora/h16_l2_seed42
```

FP16 compressed CUDA results should be checked separately because feature
compression can change probabilities and, in some datasets, final predictions.


Parameters used for the reported results
----------------------------------------

The experimental results reported in the documentation were obtained using the
following conventions:

- all implementations were executed from their implementation directory;
- all compared implementations used the same converted dataset directory;
- all compared implementations used the same fixed weight directory;
- fixed weights were generated with seed `42`;
- the message-batching implementations used batch size `4096`;
- dense adjacency experiments used the configured dense-memory limit and were
  run only when the `N x N` matrix was feasible;
- `inference_ms` excludes dataset loading, weight loading, preprocessing,
  initial CUDA H2D transfers, and final CUDA D2H transfers;
- compiled executables are generated artifacts and do not need to be included in
  the delivered source directory.
