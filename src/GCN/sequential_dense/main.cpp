#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../utilities/graph.h"
#include "../utilities/inference.h"

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMilliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

bool checkedMultiply(size_t lhs, size_t rhs, size_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

void printUsage(const char* executable) {
    std::cerr << "Uso: " << executable
              << " <nome_dataset> <hidden_dim> <num_classes> <num_layers> [max_dense_memory_mb]"
              << std::endl;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Errore: Parametri mancanti." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);
    const size_t max_dense_memory_mb = argc > 5
        ? static_cast<size_t>(std::stoull(argv[5]))
        : 1024ULL;

    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1) {
        std::cerr << "Errore: hidden_dim, num_classes e num_layers devono essere positivi."
                  << std::endl;
        return 1;
    }

    const std::string folder_path = "../../../dataset/converted/" + dataset_name;
    Graph graph;

    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes << std::endl;
    std::cout << "Modalita: Sequential Dense" << std::endl;
    std::cout << "Percorso: " << folder_path << std::endl;
    std::cout << "========================================" << std::endl;

    if (!graph.loadGraph(folder_path)) {
        std::cerr << "Errore nel caricamento del grafo!" << std::endl;
        return 1;
    }

    const int num_nodes = graph.getNumNodes();
    const int num_edges = graph.getNumEdges();
    const int feature_dim = graph.getFeatureDim();

    size_t adjacency_elements = 0;
    if (!checkedMultiply(static_cast<size_t>(num_nodes), static_cast<size_t>(num_nodes),
                         adjacency_elements)) {
        std::cerr << "Errore: overflow nel calcolo della matrice densa." << std::endl;
        return 1;
    }

    size_t adjacency_bytes = 0;
    if (!checkedMultiply(adjacency_elements, sizeof(float), adjacency_bytes)) {
        std::cerr << "Errore: overflow nel calcolo della memoria densa." << std::endl;
        return 1;
    }

    const size_t max_dense_bytes = max_dense_memory_mb * 1024ULL * 1024ULL;
    if (adjacency_bytes > max_dense_bytes) {
        std::cerr << "Dense non eseguibile: la matrice di adiacenza richiede circa "
                  << (static_cast<double>(adjacency_bytes) / (1024.0 * 1024.0))
                  << " MB, limite configurato " << max_dense_memory_mb << " MB."
                  << std::endl;
        return 2;
    }

    std::cout << std::fixed << std::setprecision(2)
              << "Memoria adiacenza densa: "
              << (static_cast<double>(adjacency_bytes) / (1024.0 * 1024.0))
              << " MB" << std::endl;

    std::vector<float> adjacency(adjacency_elements, 0.0f);
    std::vector<int> in_degree(num_nodes, 0);

    const std::vector<int>& row_pointers = graph.getRowPointers();
    const std::vector<int>& column_indices = graph.getColumnIndices();
    for (int src = 0; src < num_nodes; ++src) {
        for (int edge = row_pointers[src]; edge < row_pointers[src + 1]; ++edge) {
            const int dest = column_indices[edge];
            adjacency[static_cast<size_t>(src) * num_nodes + dest] = 1.0f;
            ++in_degree[dest];
        }
    }

    std::vector<LayerWeights> weights;
    if (num_layers == 1) {
        weights.emplace_back(feature_dim, num_classes);
    } else {
        weights.emplace_back(feature_dim, hidden_dim);
        for (int layer = 1; layer < num_layers - 1; ++layer) {
            weights.emplace_back(hidden_dim, hidden_dim);
        }
        weights.emplace_back(hidden_dim, num_classes);
    }

    std::vector<std::vector<float>> h_current(num_nodes);
    for (int vertex = 0; vertex < num_nodes; ++vertex) {
        h_current[vertex] = graph.getVertex(vertex).features;
    }

    std::cout << "Inizio elaborazione inference dense..." << std::endl;
    const auto inference_begin = Clock::now();

    int current_dim = feature_dim;
    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        std::vector<std::vector<float>> h_next(num_nodes,
                                               std::vector<float>(next_dim, 0.0f));

        for (int dest = 0; dest < num_nodes; ++dest) {
            std::vector<float> message(current_dim, 0.0f);

            if (in_degree[dest] > 0) {
                for (int src = 0; src < num_nodes; ++src) {
                    if (adjacency[static_cast<size_t>(src) * num_nodes + dest] == 0.0f) {
                        continue;
                    }
                    for (int feature = 0; feature < current_dim; ++feature) {
                        message[feature] += h_current[src][feature];
                    }
                }

                const float inverse_degree = 1.0f / static_cast<float>(in_degree[dest]);
                for (float& value : message) {
                    value *= inverse_degree;
                }
            } else {
                message = h_current[dest];
            }

            for (int output = 0; output < next_dim; ++output) {
                float dot_product = 0.0f;
                const size_t weight_offset = static_cast<size_t>(output) * current_dim;
                for (int feature = 0; feature < current_dim; ++feature) {
                    dot_product += weights[layer].W[weight_offset + feature] *
                                   message[feature];
                }
                h_next[dest][output] = relu(dot_product);
            }
        }

        h_current = std::move(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << "/" << num_layers
                  << " completato." << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;
    for (int vertex = 0; vertex < num_nodes; ++vertex) {
        float max_logit = h_current[vertex][0];
        for (int class_id = 1; class_id < num_classes; ++class_id) {
            max_logit = std::max(max_logit, h_current[vertex][class_id]);
        }

        float sum_exp = 0.0f;
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            h_current[vertex][class_id] =
                std::exp(h_current[vertex][class_id] - max_logit);
            sum_exp += h_current[vertex][class_id];
        }
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            h_current[vertex][class_id] /= sum_exp;
        }
    }

    const auto inference_end = Clock::now();
    const double inference_ms = elapsedMilliseconds(inference_begin, inference_end);
    const double seconds = inference_ms / 1000.0;
    const double nodes_per_second = seconds > 0.0 ? num_nodes / seconds : 0.0;
    const double messages_per_second = seconds > 0.0
        ? static_cast<double>(num_edges) * num_layers / seconds
        : 0.0;

    std::cout << std::fixed << std::setprecision(6)
              << "RESULT implementation=sequential-dense"
              << " inference_ms=" << inference_ms
              << " nodes_per_second=" << nodes_per_second
              << " messages_per_second=" << messages_per_second
              << " dense_adjacency_mb="
              << (static_cast<double>(adjacency_bytes) / (1024.0 * 1024.0))
              << std::endl;

    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}
