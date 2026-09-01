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

inline double elapsedMilliseconds(BenchmarkClock::time_point begin,
                                  BenchmarkClock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

inline void reportResults(const std::string& implementation,
                          const std::vector<float>& probabilities,
                          const std::vector<int>& labels,
                          int num_nodes,
                          int num_edges,
                          int num_classes,
                          int num_layers,
                          double inference_ms) {
    std::uint64_t prediction_checksum = 1469598103934665603ULL;
    double probability_checksum = 0.0;
    int correct = 0;
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
        if (node < static_cast<int>(labels.size()) && prediction == labels[node]) ++correct;
    }

    const double seconds = inference_ms / 1000.0;
    const double nodes_per_second = seconds > 0.0 ? num_nodes / seconds : 0.0;
    const double edges_per_second = seconds > 0.0
        ? static_cast<double>(num_edges) * num_layers / seconds : 0.0;
    const double accuracy = labels.size() >= static_cast<size_t>(num_nodes)
        ? static_cast<double>(correct) / num_nodes : -1.0;

    std::cout << std::fixed << std::setprecision(6)
              << "RESULT implementation=" << implementation
              << " inference_ms=" << inference_ms
              << " nodes_per_second=" << nodes_per_second
              << " messages_per_second=" << edges_per_second
              << " probability_checksum=" << probability_checksum
              << " prediction_checksum=" << prediction_checksum
              << " accuracy=" << accuracy << '\n';

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
