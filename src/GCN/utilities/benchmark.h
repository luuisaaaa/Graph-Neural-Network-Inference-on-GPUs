#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
                          int num_nodes,
                          int num_edges,
                          int num_classes,
                          int num_layers,
                          double inference_ms) {
        
const double seconds = inference_ms / 1000.0;

const long long total_nodes = static_cast<long long>(num_nodes) * num_layers;
const long long total_edges = static_cast<long long>(num_edges) * num_layers;

const double nodes_per_second = (seconds > 0.0) ? (static_cast<double>(total_nodes) / seconds) : 0.0;
const double edges_per_second = (seconds > 0.0) ? (static_cast<double>(total_edges) / seconds) : 0.0;

    std::ofstream outfile("./result.txt", std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Errore: impossibile aprire ./result.txt per la scrittura." << std::endl;
        return;
    }

    outfile << "========================================\n"
            << "RESULT implementation:" << implementation << "\n"
            << std::fixed
            << std::setprecision(2)
            << "Tempo:" << inference_ms << " ms\n"
            << std::setprecision(0)
            << "Nodi per secondo:" << nodes_per_second << " nodes/s\n"
            << "Archi per secondo:" << edges_per_second << " edges/s\n";

    outfile.close();
}

#endif