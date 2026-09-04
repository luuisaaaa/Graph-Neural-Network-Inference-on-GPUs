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
void checkCuda(cudaError_t error) {
    if (error != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(error) << '\n';
        std::exit(EXIT_FAILURE);
    }
}

// Un thread elabora un messaggio del batch.
__global__ void aggregate_batch_kernel(const int* sources, const int* destinations,
                                       const float* h_current, float* aggregated,
                                       int batch_begin, int batch_count, int feature_dim) {
    const int local_message = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_message >= batch_count) return;
    const int message = batch_begin + local_message;
    const int source = sources[message];
    const int destination = destinations[message];
    for (int feature = 0; feature < feature_dim; ++feature) {
        atomicAdd(&aggregated[destination * feature_dim + feature],
                  h_current[source * feature_dim + feature]);
    }
}

__global__ void update_kernel(const float* h_current, float* aggregated, const float* weights,
                              float* h_next, const int* degree, int num_nodes,
                              int current_dim, int next_dim) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= num_nodes) return;
    const int msg = vertex * current_dim;
    for (int feature = 0; feature < current_dim; ++feature) {
        aggregated[msg + feature] = degree[vertex] > 0
            ? aggregated[msg + feature] / static_cast<float>(degree[vertex])
            : h_current[msg + feature];
    }
    for (int output = 0; output < next_dim; ++output) {
        float dot = 0.0f;
        for (int feature = 0; feature < current_dim; ++feature)
            dot += weights[output * current_dim + feature] * aggregated[msg + feature];
        h_next[vertex * next_dim + output] = fmaxf(0.0f, dot);
    }
}

__global__ void softmax_kernel(float* values, int num_nodes, int num_classes) {
    const int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= num_nodes) return;
    const int offset = vertex * num_classes;
    float maximum = values[offset];
    for (int c = 1; c < num_classes; ++c) maximum = fmaxf(maximum, values[offset + c]);
    float sum = 0.0f;
    for (int c = 0; c < num_classes; ++c) {
        values[offset + c] = expf(values[offset + c] - maximum);
        sum += values[offset + c];
    }
    for (int c = 0; c < num_classes; ++c) values[offset + c] /= sum;
}
}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Uso: " << argv[0]
                  << " <nome_dataset> <hidden_dim> <num_classes> <num_layers>"
                     " <cartella_pesi> [message_batch_size]\n";
        return 1;
    }
    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);
    const std::string weights_path = argv[5];
    const int batch_size = argc > 6 ? std::stoi(argv[6]) : 4096;
    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1 || batch_size < 1) return 1;

    int device_count = 0;
    checkCuda(cudaGetDeviceCount(&device_count));
    if (device_count == 0) {
        std::cerr << "Errore: nessuna GPU CUDA rilevata.\n";
        return 1;
    }
    int device = 0;
    checkCuda(cudaGetDevice(&device));
    const std::string folder_path = "../../../../../dataset/converted/" + dataset_name;
    Graph graph;
    std::cout << "========================================\nDataset: " << dataset_name
              << "\nLayer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes
              << "\nModalita: CUDA Pure Message-Batching Basic (1 Thread per Messaggio)"
              << "\nMessaggi per batch: " << batch_size << "\nGPU: " << device
              << "\nPercorso: " << folder_path << "\n========================================\n";
    if (!graph.loadGraph(folder_path)) return 1;

    const int num_nodes = graph.getNumNodes();
    const int num_edges = graph.getNumEdges();
    const int feature_dim = graph.getFeatureDim();
    const auto& rows = graph.getRowPointers();
    const auto& host_sources = graph.getColumnIndices();
    std::vector<int> host_destinations(num_edges), host_degree(num_nodes);
    for (int vertex = 0; vertex < num_nodes; ++vertex) {
        host_degree[vertex] = rows[vertex + 1] - rows[vertex];
        for (int message = rows[vertex]; message < rows[vertex + 1]; ++message)
            host_destinations[message] = vertex;
    }

    std::vector<int> layer_dimensions{feature_dim};
    for (int layer = 1; layer < num_layers; ++layer) layer_dimensions.push_back(hidden_dim);
    layer_dimensions.push_back(num_classes);
    std::vector<LayerWeights> weights;
    std::string weights_error;
    if (!loadModelWeights(weights_path, dataset_name, layer_dimensions, weights,
                          weights_error)) {
        std::cerr << "Errore nel caricamento dei pesi: " << weights_error << '\n';
        return 1;
    }
    std::cout << "Pesi precaricati da: " << weights_path << '\n';
    std::vector<size_t> offsets(num_layers);
    size_t total_weights = 0;
    for (int l = 0; l < num_layers; ++l) {
        offsets[l] = total_weights;
        total_weights += weights[l].W.size();
    }
    std::vector<float> host_weights(total_weights);
    for (int l = 0; l < num_layers; ++l)
        std::copy(weights[l].W.begin(), weights[l].W.end(), host_weights.begin() + offsets[l]);

    int *d_sources = nullptr, *d_destinations = nullptr, *d_degree = nullptr;
    float *d_current = nullptr, *d_next = nullptr, *d_aggregated = nullptr, *d_weights = nullptr;
    const int max_dim = std::max({feature_dim, hidden_dim, num_classes});
    checkCuda(cudaMalloc(&d_sources, num_edges * sizeof(int)));
    checkCuda(cudaMalloc(&d_destinations, num_edges * sizeof(int)));
    checkCuda(cudaMalloc(&d_degree, num_nodes * sizeof(int)));
    checkCuda(cudaMalloc(&d_current, static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCuda(cudaMalloc(&d_next, static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCuda(cudaMalloc(&d_aggregated, static_cast<size_t>(num_nodes) * max_dim * sizeof(float)));
    checkCuda(cudaMalloc(&d_weights, total_weights * sizeof(float)));
    checkCuda(cudaMemcpy(d_sources, host_sources.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_destinations, host_destinations.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_degree, host_degree.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_current, graph.getNodeFeatures().data(), static_cast<size_t>(num_nodes) * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weights, host_weights.data(), total_weights * sizeof(float), cudaMemcpyHostToDevice));

    constexpr int threads = 256;
    const int node_blocks = (num_nodes + threads - 1) / threads;
    int current_dim = feature_dim;
    cudaEvent_t inference_begin, inference_end;
    checkCuda(cudaEventCreate(&inference_begin));
    checkCuda(cudaEventCreate(&inference_end));
    checkCuda(cudaEventRecord(inference_begin));
    std::cout << "Inizio elaborazione inference CUDA...\n";
    for (int l = 0; l < num_layers; ++l) {
        const int next_dim = weights[l].out_dim;
        checkCuda(cudaMemset(d_aggregated, 0, static_cast<size_t>(num_nodes) * current_dim * sizeof(float)));
        for (int begin = 0; begin < num_edges; begin += batch_size) {
            const int count = std::min(batch_size, num_edges - begin);
            aggregate_batch_kernel<<<(count + threads - 1) / threads, threads>>>(
                d_sources, d_destinations, d_current, d_aggregated, begin, count, current_dim);
            checkCuda(cudaGetLastError());
        }
        update_kernel<<<node_blocks, threads>>>(d_current, d_aggregated, d_weights + offsets[l],
                                                d_next, d_degree, num_nodes, current_dim, next_dim);
        checkCuda(cudaGetLastError());
        std::swap(d_current, d_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << l + 1 << "/" << num_layers << " completato ("
                  << (num_edges + batch_size - 1) / batch_size << " batch di messaggi).\n";
    }
    softmax_kernel<<<node_blocks, threads>>>(d_current, num_nodes, num_classes);
    checkCuda(cudaGetLastError());
    checkCuda(cudaEventRecord(inference_end));
    checkCuda(cudaEventSynchronize(inference_end));
    float inference_ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&inference_ms, inference_begin, inference_end));
    std::vector<float> output(static_cast<size_t>(num_nodes) * num_classes);
    checkCuda(cudaMemcpy(output.data(), d_current, output.size() * sizeof(float), cudaMemcpyDeviceToHost));
    reportResults("cuda-message-batching-basic", output, graph.getLabels(), num_nodes, num_edges,
                  num_classes, num_layers, inference_ms);
    checkCuda(cudaEventDestroy(inference_begin));
    checkCuda(cudaEventDestroy(inference_end));
    checkCuda(cudaFree(d_sources)); checkCuda(cudaFree(d_destinations)); checkCuda(cudaFree(d_degree));
    checkCuda(cudaFree(d_current)); checkCuda(cudaFree(d_next)); checkCuda(cudaFree(d_aggregated));
    checkCuda(cudaFree(d_weights));
    std::cout << "Elaborazione conclusa con successo!\n";
    return 0;
}
