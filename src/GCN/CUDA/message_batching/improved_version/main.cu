#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../../../utilities/graph.h"
#include "../../../../utilities/inference.h"

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

// Un warp per vertice del batch; i lane elaborano output diversi. I pesi sono
// trasposti per ottenere letture contigue tra i lane dello stesso warp.
__global__ void message_batch_kernel(const int* row_pointers,
                                     const int* column_indices,
                                     const float* h_current,
                                     const float* weights_transposed,
                                     float* h_next,
                                     int batch_begin,
                                     int batch_count,
                                     int current_dim,
                                     int next_dim) {
    const int lane = threadIdx.x;
    const int local_vertex = blockIdx.x * blockDim.y + threadIdx.y;
    if (local_vertex >= batch_count) return;
    const int vertex = batch_begin + local_vertex;
    const int edge_begin = row_pointers[vertex];
    const int edge_end = row_pointers[vertex + 1];
    const int degree = edge_end - edge_begin;

    for (int output = lane; output < next_dim; output += kWarpSize) {
        float dot_product = 0.0f;
        for (int feature = 0; feature < current_dim; ++feature) {
            float message = 0.0f;
            if (degree > 0) {
                for (int edge = edge_begin; edge < edge_end; ++edge) {
                    message += h_current[column_indices[edge] * current_dim + feature];
                }
                message /= static_cast<float>(degree);
            } else {
                message = h_current[vertex * current_dim + feature];
            }
            dot_product += weights_transposed[feature * next_dim + output] * message;
        }
        h_next[vertex * next_dim + output] = fmaxf(0.0f, dot_product);
    }
}

__device__ float warpMax(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warpSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__global__ void softmax_batch_kernel(float* values, int batch_begin, int batch_count,
                                     int num_classes) {
    const int lane = threadIdx.x;
    const int local_vertex = blockIdx.x * blockDim.y + threadIdx.y;
    if (local_vertex >= batch_count) return;
    const int offset = (batch_begin + local_vertex) * num_classes;

    float local_max = -CUDART_INF_F;
    for (int c = lane; c < num_classes; c += kWarpSize) {
        local_max = fmaxf(local_max, values[offset + c]);
    }
    const float max_logit = warpMax(local_max);
    float local_sum = 0.0f;
    for (int c = lane; c < num_classes; c += kWarpSize) {
        const float value = expf(values[offset + c] - max_logit);
        values[offset + c] = value;
        local_sum += value;
    }
    const float sum = warpSum(local_sum);
    for (int c = lane; c < num_classes; c += kWarpSize) values[offset + c] /= sum;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Errore: Parametri mancanti.\nUso: " << argv[0]
                  << " <nome_dataset> <hidden_dim> <num_classes> <num_layers> [batch_size]"
                  << std::endl;
        return 1;
    }
    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);
    const int batch_size = argc > 5 ? std::stoi(argv[5]) : 1024;
    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1 || batch_size < 1) {
        std::cerr << "Errore: tutti i parametri numerici devono essere positivi." << std::endl;
        return 1;
    }

    int device_count = 0;
    checkCudaError(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Errore: Nessun dispositivo GPU abilitato CUDA rilevato." << std::endl;
        return 1;
    }
    int device = 0;
    checkCudaError(cudaGetDevice(&device));
    const std::string folder_path = "../../../../../dataset/converted/" + dataset_name;
    Graph graph;
    std::cout << "========================================\nDataset: " << dataset_name
              << "\nLayer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes
              << "\nModalita: CUDA Message-Batching Improved (1 Warp per Vertice)"
              << "\nBatch size: " << batch_size << "\nDispositivo GPU CUDA in uso: " << device
              << "\nPercorso: " << folder_path << "\n========================================" << std::endl;
    if (!graph.loadGraph(folder_path)) return 1;

    const int num_nodes = graph.getNumNodes();
    const int num_edges = graph.getNumEdges();
    const int feature_dim = graph.getFeatureDim();
    std::vector<LayerWeights> weights;
    if (num_layers == 1) weights.emplace_back(feature_dim, num_classes);
    else {
        weights.emplace_back(feature_dim, hidden_dim);
        for (int layer = 1; layer < num_layers - 1; ++layer) weights.emplace_back(hidden_dim, hidden_dim);
        weights.emplace_back(hidden_dim, num_classes);
    }

    std::vector<size_t> offsets(num_layers);
    size_t total_weights = 0;
    for (int layer = 0; layer < num_layers; ++layer) {
        offsets[layer] = total_weights;
        total_weights += weights[layer].W.size();
    }
    std::vector<float> host_weights(total_weights);
    for (int layer = 0; layer < num_layers; ++layer) {
        const int in_dim = weights[layer].in_dim;
        const int out_dim = weights[layer].out_dim;
        for (int input = 0; input < in_dim; ++input) {
            for (int output = 0; output < out_dim; ++output) {
                host_weights[offsets[layer] + static_cast<size_t>(input) * out_dim + output] =
                    weights[layer].W[static_cast<size_t>(output) * in_dim + input];
            }
        }
    }

    int *d_rows = nullptr, *d_columns = nullptr;
    float *d_current = nullptr, *d_next = nullptr, *d_weights = nullptr;
    const int max_dim = std::max({feature_dim, hidden_dim, num_classes});
    checkCudaError(cudaMalloc(&d_rows, (num_nodes + 1) * sizeof(int)));
    checkCudaError(cudaMalloc(&d_columns, num_edges * sizeof(int)));
    checkCudaError(cudaMalloc(&d_current, static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCudaError(cudaMalloc(&d_next, static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCudaError(cudaMalloc(&d_weights, total_weights * sizeof(float)));
    checkCudaError(cudaMemcpy(d_rows, graph.getRowPointers().data(), (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_columns, graph.getColumnIndices().data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_current, graph.getNodeFeatures().data(), static_cast<size_t>(num_nodes) * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_weights, host_weights.data(), total_weights * sizeof(float), cudaMemcpyHostToDevice));

    const dim3 threads(kWarpSize, kWarpsPerBlock);
    int current_dim = feature_dim;
    std::cout << "Inizio elaborazione inference CUDA..." << std::endl;
    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        for (int begin = 0; begin < num_nodes; begin += batch_size) {
            const int count = std::min(batch_size, num_nodes - begin);
            message_batch_kernel<<<(count + kWarpsPerBlock - 1) / kWarpsPerBlock, threads>>>(
                d_rows, d_columns, d_current, d_weights + offsets[layer], d_next,
                begin, count, current_dim, next_dim);
            checkCudaError(cudaGetLastError());
        }
        std::swap(d_current, d_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << "/" << num_layers << " completato ("
                  << (num_nodes + batch_size - 1) / batch_size << " batch)." << std::endl;
    }
    for (int begin = 0; begin < num_nodes; begin += batch_size) {
        const int count = std::min(batch_size, num_nodes - begin);
        softmax_batch_kernel<<<(count + kWarpsPerBlock - 1) / kWarpsPerBlock, threads>>>(
            d_current, begin, count, num_classes);
        checkCudaError(cudaGetLastError());
    }
    checkCudaError(cudaDeviceSynchronize());
    std::vector<float> output(static_cast<size_t>(num_nodes) * num_classes);
    checkCudaError(cudaMemcpy(output.data(), d_current, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaError(cudaFree(d_rows));
    checkCudaError(cudaFree(d_columns));
    checkCudaError(cudaFree(d_current));
    checkCudaError(cudaFree(d_next));
    checkCudaError(cudaFree(d_weights));
    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}
