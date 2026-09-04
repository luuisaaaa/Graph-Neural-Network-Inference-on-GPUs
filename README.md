# Graph Neural Network Inference on GPUs

## Sequential dense baseline

The dense baseline builds a full `num_nodes x num_nodes` adjacency matrix from
the CSR dataset files, then runs the same GCN inference flow used by the
sequential sparse version. It is intended for sparse-vs-dense experiments on
small and medium graphs.

Compile from `src/GCN/sequential_dense`:

```bash
g++ -O2 -std=c++17 main.cpp ../utilities/graph.cpp ../utilities/inference.cpp -o sequential_dense
```

Run, for example, on Cora:

```bash
./sequential_dense Cora 16 7 2
```

An optional fifth parameter sets the maximum dense-adjacency memory in MB. The
default is `1024`. If the dense matrix would exceed the limit, the program exits
with a clear "not executable" message instead of trying to allocate it.

```bash
./sequential_dense Cora 16 7 2 2048
```
