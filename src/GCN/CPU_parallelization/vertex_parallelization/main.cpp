#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include "../../utilities/benchmark.h"
#include "../../utilities/graph.h"
#include "../../utilities/inference.h"

int main(int argc, char* argv[]) {
    if (argc < 5) {
        std::cerr << "Errore: Parametri mancanti." << std::endl;
        std::cerr << "Uso: " << argv[0]
                  << " <nome_dataset> <hidden_dim> <num_classes> <num_layers>"
                  << std::endl;
        return 1;
    }

    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);

    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1) {
        std::cerr << "Errore: hidden_dim, num_classes e num_layers devono essere positivi."
                  << std::endl;
        return 1;
    }

    const std::string folder_path = "../../../../dataset/converted/" + dataset_name;
    Graph graph;

    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes << std::endl;
    std::cout << "Modalita: Vertex-Parallel" << std::endl;
    std::cout << "Thread OpenMP massimi: " << omp_get_max_threads() << std::endl;
    std::cout << "Percorso: " << folder_path << std::endl;
    std::cout << "========================================" << std::endl;

    if (!graph.loadGraph(folder_path)) {
        std::cerr << "Errore nel caricamento del grafo!" << std::endl;
        return 1;
    }

    const int num_nodes = graph.getNumNodes();
    const int feature_dim = graph.getFeatureDim();
    const std::vector<int>& row_pointers = graph.getRowPointers();
    const std::vector<int>& column_indices = graph.getColumnIndices();

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

    std::vector<float> h_current = graph.getNodeFeatures();

    std::cout << "Inizio elaborazione inference..." << std::endl;
    const auto inference_begin = BenchmarkClock::now();
    int current_dim = feature_dim;
    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        std::vector<float> h_next(static_cast<size_t>(num_nodes) * next_dim, 0.0f);

        // Ogni iterazione possiede in esclusiva il vettore del vertice v:
        // non sono necessarie operazioni atomiche durante l'aggregazione CSR.
#pragma omp parallel for schedule(dynamic, 64)
        for (int v = 0; v < num_nodes; ++v) {
            std::vector<float> message(current_dim, 0.0f);
            const int edge_begin = row_pointers[v];
            const int edge_end = row_pointers[v + 1];
            const int degree = edge_end - edge_begin;

            if (degree > 0) {
                for (int edge = edge_begin; edge < edge_end; ++edge) {
                    const int neighbor = column_indices[edge];
                    const size_t neighbor_offset = static_cast<size_t>(neighbor) * current_dim;
                    for (int feature = 0; feature < current_dim; ++feature) {
                        message[feature] += h_current[neighbor_offset + feature];
                    }
                }
                const float inverse_degree = 1.0f / static_cast<float>(degree);
                for (float& value : message) {
                    value *= inverse_degree;
                }
            } else {
                const size_t vertex_offset = static_cast<size_t>(v) * current_dim;
                std::copy_n(h_current.begin() + vertex_offset, current_dim, message.begin());
            }

            const size_t output_offset = static_cast<size_t>(v) * next_dim;
            for (int output = 0; output < next_dim; ++output) {
                float dot_product = 0.0f;
                const size_t weight_offset = static_cast<size_t>(output) * current_dim;
                for (int feature = 0; feature < current_dim; ++feature) {
                    dot_product += weights[layer].W[weight_offset + feature] * message[feature];
                }
                h_next[output_offset + output] = relu(dot_product);
            }
        }

        h_current.swap(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << "/" << num_layers << " completato."
                  << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;
#pragma omp parallel for schedule(static)
    for (int v = 0; v < num_nodes; ++v) {
        const size_t offset = static_cast<size_t>(v) * num_classes;
        float max_logit = h_current[offset];
        for (int class_id = 1; class_id < num_classes; ++class_id) {
            max_logit = std::max(max_logit, h_current[offset + class_id]);
        }

        float sum_exp = 0.0f;
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            h_current[offset + class_id] = std::exp(h_current[offset + class_id] - max_logit);
            sum_exp += h_current[offset + class_id];
        }
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            h_current[offset + class_id] /= sum_exp;
        }
    }

    const auto inference_end = BenchmarkClock::now();
    reportResults("cpu-vertex-parallel", h_current, graph.getLabels(), num_nodes,
                  graph.getNumEdges(), num_classes, num_layers,
                  elapsedMilliseconds(inference_begin, inference_end));

    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}
