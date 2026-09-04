#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../../../utilities/benchmark.h"
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

// Un warp elabora un vertice. I lane collaborano sulle feature e sugli output.
// I pesi sono trasposti sull'host per rendere contigue le letture dei lane.
__global__ void vertex_layer_kernel(const int* row_pointers,
                                    const int* column_indices,
                                    const float* h_current,
                                    const float* weights_transposed,
                                    float* h_next,
                                    int num_nodes,
                                    int current_dim,
                                    int next_dim) {
    const int lane = threadIdx.x;
    const int vertex = blockIdx.x * blockDim.y + threadIdx.y;
    if (vertex >= num_nodes) {
        return;
    }

    const int edge_begin = row_pointers[vertex];
    const int edge_end = row_pointers[vertex + 1];
    const int degree = edge_end - edge_begin;

    for (int output = lane; output < next_dim; output += kWarpSize) {
        float dot_product = 0.0f;
        for (int feature = 0; feature < current_dim; ++feature) {
            float aggregate = 0.0f;
            if (degree > 0) {
                for (int edge = edge_begin; edge < edge_end; ++edge) {
                    const int neighbor = column_indices[edge];
                    aggregate += h_current[neighbor * current_dim + feature];
                }
                aggregate /= static_cast<float>(degree);
            } else {
                aggregate = h_current[vertex * current_dim + feature];
            }
            dot_product += weights_transposed[feature * next_dim + output] * aggregate;
        }
        h_next[vertex * next_dim + output] = fmaxf(0.0f, dot_product);
    }
}

__device__ float warpReduceMax(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__device__ float warpReduceSum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return __shfl_sync(0xffffffff, value, 0);
}

__global__ void softmax_kernel(float* values, int num_nodes, int num_classes) {
    const int lane = threadIdx.x;
    const int vertex = blockIdx.x * blockDim.y + threadIdx.y;
    if (vertex >= num_nodes) {
        return;
    }

    const int offset = vertex * num_classes;
    float local_max = -CUDART_INF_F;
    for (int class_id = lane; class_id < num_classes; class_id += kWarpSize) {
        local_max = fmaxf(local_max, values[offset + class_id]);
    }
    const float max_logit = warpReduceMax(local_max);

    float local_sum = 0.0f;
    for (int class_id = lane; class_id < num_classes; class_id += kWarpSize) {
        const float value = expf(values[offset + class_id] - max_logit);
        values[offset + class_id] = value;
        local_sum += value;
    }
    const float sum_exp = warpReduceSum(local_sum);
    for (int class_id = lane; class_id < num_classes; class_id += kWarpSize) {
        values[offset + class_id] /= sum_exp;
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Errore: Parametri mancanti." << std::endl;
        std::cerr << "Uso: " << argv[0]
                  << " <nome_dataset> <hidden_dim> <num_classes> <num_layers> <cartella_pesi>"
                  << std::endl;
        return 1;
    }

    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);
    const std::string weights_path = argv[5];
    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1) {
        std::cerr << "Errore: hidden_dim, num_classes e num_layers devono essere positivi."
                  << std::endl;
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
    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes << std::endl;
    std::cout << "Modalita: CUDA Vertex-Parallel (1 Warp per Vertice)" << std::endl;
    std::cout << "Dispositivo GPU CUDA in uso: " << device << std::endl;
    std::cout << "Percorso: " << folder_path << std::endl;
    std::cout << "========================================" << std::endl;

    if (!graph.loadGraph(folder_path)) {
        std::cerr << "Errore nel caricamento del grafo!" << std::endl;
        return 1;
    }

    const int num_nodes = graph.getNumNodes();
    const int num_edges = graph.getNumEdges();
    const int feature_dim = graph.getFeatureDim();

    std::vector<int> layer_dimensions{feature_dim};
    for (int layer = 1; layer < num_layers; ++layer) layer_dimensions.push_back(hidden_dim);
    layer_dimensions.push_back(num_classes);
    std::vector<LayerWeights> weights;
    std::string weights_error;
    if (!loadModelWeights(weights_path, dataset_name, layer_dimensions, weights,
                          weights_error)) {
        std::cerr << "Errore nel caricamento dei pesi: " << weights_error << std::endl;
        return 1;
    }
    std::cout << "Pesi precaricati da: " << weights_path << std::endl;

    std::vector<size_t> weight_offsets(num_layers);
    size_t total_weights = 0;
    for (int layer = 0; layer < num_layers; ++layer) {
        weight_offsets[layer] = total_weights;
        total_weights += weights[layer].W.size();
    }
    std::vector<float> host_weights(total_weights);
    for (int layer = 0; layer < num_layers; ++layer) {
        const int current_dim = weights[layer].in_dim;
        const int next_dim = weights[layer].out_dim;
        const size_t offset = weight_offsets[layer];
        for (int input = 0; input < current_dim; ++input) {
            for (int output = 0; output < next_dim; ++output) {
                host_weights[offset + static_cast<size_t>(input) * next_dim + output] =
                    weights[layer].W[static_cast<size_t>(output) * current_dim + input];
            }
        }
    }

    int* device_row_pointers = nullptr;
    int* device_column_indices = nullptr;
    float* device_current = nullptr;
    float* device_next = nullptr;
    float* device_weights = nullptr;
    const int max_dim = std::max({feature_dim, hidden_dim, num_classes});

    checkCudaError(cudaMalloc(&device_row_pointers, (num_nodes + 1) * sizeof(int)));
    checkCudaError(cudaMalloc(&device_column_indices, num_edges * sizeof(int)));
    checkCudaError(cudaMalloc(&device_current,
                              static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCudaError(cudaMalloc(&device_next,
                              static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCudaError(cudaMalloc(&device_weights, total_weights * sizeof(float)));

    checkCudaError(cudaMemcpy(device_row_pointers, graph.getRowPointers().data(),
                              (num_nodes + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(device_column_indices, graph.getColumnIndices().data(),
                              num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(device_current, graph.getNodeFeatures().data(),
                              static_cast<size_t>(num_nodes) * feature_dim * sizeof(float),
                              cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(device_weights, host_weights.data(), total_weights * sizeof(float),
                              cudaMemcpyHostToDevice));

    const dim3 threads_per_block(kWarpSize, kWarpsPerBlock);
    const dim3 blocks_per_grid((num_nodes + kWarpsPerBlock - 1) / kWarpsPerBlock);
    int current_dim = feature_dim;
    cudaEvent_t inference_begin, inference_end;
    checkCudaError(cudaEventCreate(&inference_begin));
    checkCudaError(cudaEventCreate(&inference_end));
    checkCudaError(cudaEventRecord(inference_begin));
    std::cout << "Inizio elaborazione inference CUDA..." << std::endl;
    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        vertex_layer_kernel<<<blocks_per_grid, threads_per_block>>>(
            device_row_pointers, device_column_indices, device_current,
            device_weights + weight_offsets[layer], device_next, num_nodes, current_dim, next_dim);
        checkCudaError(cudaGetLastError());
        std::swap(device_current, device_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << "/" << num_layers << " completato."
                  << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;
    softmax_kernel<<<blocks_per_grid, threads_per_block>>>(device_current, num_nodes, num_classes);
    checkCudaError(cudaGetLastError());
    checkCudaError(cudaEventRecord(inference_end));
    checkCudaError(cudaEventSynchronize(inference_end));
    float inference_ms = 0.0f;
    checkCudaError(cudaEventElapsedTime(&inference_ms, inference_begin, inference_end));

    std::vector<float> output(static_cast<size_t>(num_nodes) * num_classes);
    checkCudaError(cudaMemcpy(output.data(), device_current, output.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    reportResults("cuda-vertex-improved", output, graph.getLabels(), num_nodes, num_edges,
                  num_classes, num_layers, inference_ms);
    checkCudaError(cudaEventDestroy(inference_begin));
    checkCudaError(cudaEventDestroy(inference_end));

    checkCudaError(cudaFree(device_row_pointers));
    checkCudaError(cudaFree(device_column_indices));
    checkCudaError(cudaFree(device_current));
    checkCudaError(cudaFree(device_next));
    checkCudaError(cudaFree(device_weights));

    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}
