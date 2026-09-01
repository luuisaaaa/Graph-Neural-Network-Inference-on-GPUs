#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../../../../utilities/graph.h"
#include "../../../../utilities/inference.h"

namespace {

void checkCudaError(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(error) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

__global__ void message_batch_kernel(const int* row_pointers,
                                     const int* column_indices,
                                     const float* h_current,
                                     const float* weights,
                                     float* h_next,
                                     int batch_begin,
                                     int batch_count,
                                     int current_dim,
                                     int next_dim) {
    const int local_vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_vertex >= batch_count) return;
    const int vertex = batch_begin + local_vertex;
    const int edge_begin = row_pointers[vertex];
    const int edge_end = row_pointers[vertex + 1];
    const int degree = edge_end - edge_begin;

    for (int output = 0; output < next_dim; ++output) {
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
            dot_product += weights[output * current_dim + feature] * message;
        }
        h_next[vertex * next_dim + output] = fmaxf(0.0f, dot_product);
    }
}

__global__ void softmax_batch_kernel(float* values, int batch_begin, int batch_count,
                                     int num_classes) {
    const int local_vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_vertex >= batch_count) return;
    const int offset = (batch_begin + local_vertex) * num_classes;
    float max_logit = values[offset];
    for (int c = 1; c < num_classes; ++c) max_logit = fmaxf(max_logit, values[offset + c]);
    float sum = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        values[offset + c] = expf(values[offset + c] - max_logit);
        sum += values[offset + c];
    }
    for (int c = 0; c < num_classes; ++c) values[offset + c] /= sum;
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
              << "\nModalita: CUDA Message-Batching Basic (1 Thread per Vertice)"
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
        std::copy(weights[layer].W.begin(), weights[layer].W.end(), host_weights.begin() + offsets[layer]);
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

    constexpr int threads = 256;
    int current_dim = feature_dim;
    std::cout << "Inizio elaborazione inference CUDA..." << std::endl;
    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        for (int begin = 0; begin < num_nodes; begin += batch_size) {
            const int count = std::min(batch_size, num_nodes - begin);
            message_batch_kernel<<<(count + threads - 1) / threads, threads>>>(
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
        softmax_batch_kernel<<<(count + threads - 1) / threads, threads>>>(d_current, begin, count, num_classes);
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
