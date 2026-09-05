# Graph Neural Network Inference on GPUs

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
