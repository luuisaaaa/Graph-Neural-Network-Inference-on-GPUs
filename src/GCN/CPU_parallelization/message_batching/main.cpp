#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <omp.h>

#include "../../utilities/graph.h"
#include "../../utilities/inference.h"

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

    const std::string folder_path = "../../../../dataset/converted/" + dataset_name;
    Graph graph;
    std::cout << "========================================\nDataset: " << dataset_name
              << "\nLayer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes << "\nModalita: CPU Message-Batching"
              << "\nBatch size: " << batch_size
              << "\nThread OpenMP massimi: " << omp_get_max_threads()
              << "\nPercorso: " << folder_path
              << "\n========================================" << std::endl;

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
    int current_dim = feature_dim;
    std::cout << "Inizio elaborazione inference..." << std::endl;

    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        std::vector<float> h_next(static_cast<size_t>(num_nodes) * next_dim, 0.0f);

        // Un layer resta una singola iterazione full-batch. I mini-batch dividono
        // soltanto il lavoro: tutti leggono h_current e scrivono regioni disgiunte di h_next.
        for (int batch_begin = 0; batch_begin < num_nodes; batch_begin += batch_size) {
            const int batch_end = std::min(batch_begin + batch_size, num_nodes);
#pragma omp parallel for schedule(dynamic, 64)
            for (int vertex = batch_begin; vertex < batch_end; ++vertex) {
                std::vector<float> message(current_dim, 0.0f);
                const int edge_begin = row_pointers[vertex];
                const int edge_end = row_pointers[vertex + 1];
                const int degree = edge_end - edge_begin;

                if (degree > 0) {
                    for (int edge = edge_begin; edge < edge_end; ++edge) {
                        const int neighbor = column_indices[edge];
                        const size_t input_offset = static_cast<size_t>(neighbor) * current_dim;
                        for (int feature = 0; feature < current_dim; ++feature) {
                            message[feature] += h_current[input_offset + feature];
                        }
                    }
                    const float inverse_degree = 1.0f / static_cast<float>(degree);
                    for (float& value : message) value *= inverse_degree;
                } else {
                    const size_t input_offset = static_cast<size_t>(vertex) * current_dim;
                    std::copy_n(h_current.begin() + input_offset, current_dim, message.begin());
                }

                const size_t output_offset = static_cast<size_t>(vertex) * next_dim;
                for (int output = 0; output < next_dim; ++output) {
                    float dot_product = 0.0f;
                    const size_t weight_offset = static_cast<size_t>(output) * current_dim;
                    for (int feature = 0; feature < current_dim; ++feature) {
                        dot_product += weights[layer].W[weight_offset + feature] * message[feature];
                    }
                    h_next[output_offset + output] = relu(dot_product);
                }
            }
        }

        h_current.swap(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << "/" << num_layers << " completato ("
                  << (num_nodes + batch_size - 1) / batch_size << " batch)." << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;
    for (int batch_begin = 0; batch_begin < num_nodes; batch_begin += batch_size) {
        const int batch_end = std::min(batch_begin + batch_size, num_nodes);
#pragma omp parallel for schedule(static)
        for (int vertex = batch_begin; vertex < batch_end; ++vertex) {
            const size_t offset = static_cast<size_t>(vertex) * num_classes;
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
    }

    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}
