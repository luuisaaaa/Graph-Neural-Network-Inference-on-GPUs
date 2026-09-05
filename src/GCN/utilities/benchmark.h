#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using BenchmarkClock = std::chrono::steady_clock;

struct MemoryMetrics {
    std::uint64_t topology_bytes = 0;
    std::uint64_t feature_bytes = 0;
    std::uint64_t label_bytes = 0;
    std::uint64_t weight_bytes = 0;
    std::uint64_t working_bytes = 0;
    std::uint64_t device_bytes = 0;

    std::uint64_t totalBytes() const {
        return topology_bytes + feature_bytes + label_bytes + weight_bytes +
               working_bytes + device_bytes;
    }
};

inline MemoryMetrics makeMemoryMetrics(int num_nodes,
                                       int num_edges,
                                       int feature_dim,
                                       std::uint64_t weight_elements,
                                       std::uint64_t working_bytes,
                                       std::uint64_t device_bytes = 0) {
    MemoryMetrics memory;
    memory.topology_bytes =
        (static_cast<std::uint64_t>(num_nodes) + 1 + num_edges) * sizeof(int);
    memory.feature_bytes = static_cast<std::uint64_t>(num_nodes) * feature_dim * sizeof(float);
    memory.label_bytes = static_cast<std::uint64_t>(num_nodes) * sizeof(int);
    memory.weight_bytes = weight_elements * sizeof(float);
    memory.working_bytes = working_bytes;
    memory.device_bytes = device_bytes;
    return memory;
}

inline double elapsedMilliseconds(BenchmarkClock::time_point begin,
                                  BenchmarkClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void reportResults(const std::string& implementation,
                          const std::vector<float>& probabilities,
                          const std::vector<int>&,
                          int num_nodes,
                          int num_edges,
                          int num_classes,
                          int num_layers,
                          double inference_ms,
                          const MemoryMetrics& memory = {}) {
    std::uint64_t prediction_checksum = 1469598103934665603ULL;
    double probability_checksum = 0.0;
    for (int node = 0; node < num_nodes; ++node) {
        const size_t offset = static_cast<size_t>(node) * num_classes;
        int prediction = 0;
        for (int c = 0; c < num_classes; ++c) {
            probability_checksum += probabilities[offset + c] *
                                    static_cast<double>((node + 1) * (c + 1));
            if (probabilities[offset + c] > probabilities[offset + prediction]) prediction = c;
        }
        prediction_checksum ^= static_cast<std::uint64_t>(prediction + 1);
        prediction_checksum *= 1099511628211ULL;
    }

    const double seconds = inference_ms / 1000.0;
    const double graph_nodes_per_second = seconds > 0.0 ? num_nodes / seconds : 0.0;
    const double node_updates_per_second = seconds > 0.0
        ? static_cast<double>(num_nodes) * num_layers / seconds : 0.0;
    const double messages_per_second = seconds > 0.0
        ? static_cast<double>(num_edges) * num_layers / seconds : 0.0;

    std::cout << std::fixed << std::setprecision(6)
              << "RESULT implementation=" << implementation
              << " inference_ms=" << inference_ms
              << " graph_nodes_per_second=" << graph_nodes_per_second
              << " node_updates_per_second=" << node_updates_per_second
              << " messages_per_second=" << messages_per_second
              << " topology_memory_bytes=" << memory.topology_bytes
              << " feature_memory_bytes=" << memory.feature_bytes
              << " label_memory_bytes=" << memory.label_bytes
              << " weight_memory_bytes=" << memory.weight_bytes
              << " working_memory_bytes=" << memory.working_bytes
              << " device_memory_bytes=" << memory.device_bytes
              << " estimated_total_memory_bytes=" << memory.totalBytes()
              << " probability_checksum=" << probability_checksum
              << " prediction_checksum=" << prediction_checksum << '\n';

    const char* output_path = std::getenv("GCN_OUTPUT_FILE");
    if (output_path == nullptr || output_path[0] == '\0') return;
    std::ofstream output(output_path);
    if (!output) {
        std::cerr << "Errore: impossibile scrivere GCN_OUTPUT_FILE=" << output_path << '\n';
        return;
    }
    output << num_nodes << ' ' << num_classes << '\n' << std::setprecision(9);
    for (int node = 0; node < num_nodes; ++node) {
        const size_t offset = static_cast<size_t>(node) * num_classes;
        for (int c = 0; c < num_classes; ++c) {
            if (c != 0) output << ' ';
            output << probabilities[offset + c];
        }
        output << '\n';
    }
    std::cout << "Output numerico scritto in: " << output_path << '\n';
}

#endif
