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
        std::cerr << "Uso: " << argv[0]
                  << " <nome_dataset> <hidden_dim> <num_classes> <num_layers> [message_batch_size]\n";
        return 1;
    }
    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);
    const int batch_size = argc > 5 ? std::stoi(argv[5]) : 4096;
    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1 || batch_size < 1) {
        std::cerr << "Errore: tutti i parametri numerici devono essere positivi.\n";
        return 1;
    }

    const std::string folder_path = "../../../../dataset/converted/" + dataset_name;
    Graph graph;
    std::cout << "========================================\nDataset: " << dataset_name
              << "\nLayer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes << "\nModalita: CPU Pure Message-Batching"
              << "\nMessaggi per batch: " << batch_size
              << "\nThread OpenMP massimi: " << omp_get_max_threads()
              << "\nPercorso: " << folder_path
              << "\n========================================\n";
    if (!graph.loadGraph(folder_path)) return 1;

    const int num_nodes = graph.getNumNodes();
    const int num_edges = graph.getNumEdges();
    const int feature_dim = graph.getFeatureDim();
    const std::vector<int>& rows = graph.getRowPointers();
    const std::vector<int>& message_sources = graph.getColumnIndices();

    // L'elemento CSR nella riga v descrive un messaggio neighbor -> v.
    std::vector<int> message_destinations(num_edges);
    std::vector<int> degree(num_nodes);
    for (int vertex = 0; vertex < num_nodes; ++vertex) {
        degree[vertex] = rows[vertex + 1] - rows[vertex];
        for (int message = rows[vertex]; message < rows[vertex + 1]; ++message) {
            message_destinations[message] = vertex;
        }
    }

    std::vector<LayerWeights> weights;
    if (num_layers == 1) weights.emplace_back(feature_dim, num_classes);
    else {
        weights.emplace_back(feature_dim, hidden_dim);
        for (int layer = 1; layer < num_layers - 1; ++layer)
            weights.emplace_back(hidden_dim, hidden_dim);
        weights.emplace_back(hidden_dim, num_classes);
    }

    std::vector<float> h_current = graph.getNodeFeatures();
    int current_dim = feature_dim;
    std::cout << "Inizio elaborazione inference...\n";
    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        std::vector<float> aggregated(static_cast<size_t>(num_nodes) * current_dim, 0.0f);
        std::vector<float> h_next(static_cast<size_t>(num_nodes) * next_dim, 0.0f);

        // Il buffer aggregated persiste tra i batch: ogni batch contiene archi/messaggi.
        for (int batch_begin = 0; batch_begin < num_edges; batch_begin += batch_size) {
            const int batch_end = std::min(batch_begin + batch_size, num_edges);
#pragma omp parallel for schedule(static)
            for (int message = batch_begin; message < batch_end; ++message) {
                const int source = message_sources[message];
                const int destination = message_destinations[message];
                const size_t src = static_cast<size_t>(source) * current_dim;
                const size_t dst = static_cast<size_t>(destination) * current_dim;
                for (int feature = 0; feature < current_dim; ++feature) {
#pragma omp atomic update
                    aggregated[dst + feature] += h_current[src + feature];
                }
            }
        }

        // Solo dopo tutti i batch si eseguono media, trasformazione e ReLU.
#pragma omp parallel for schedule(dynamic, 64)
        for (int vertex = 0; vertex < num_nodes; ++vertex) {
            const size_t msg = static_cast<size_t>(vertex) * current_dim;
            if (degree[vertex] > 0) {
                const float inverse_degree = 1.0f / static_cast<float>(degree[vertex]);
                for (int feature = 0; feature < current_dim; ++feature)
                    aggregated[msg + feature] *= inverse_degree;
            } else {
                std::copy_n(h_current.begin() + msg, current_dim, aggregated.begin() + msg);
            }

            const size_t out = static_cast<size_t>(vertex) * next_dim;
            for (int output = 0; output < next_dim; ++output) {
                float dot = 0.0f;
                const size_t w = static_cast<size_t>(output) * current_dim;
                for (int feature = 0; feature < current_dim; ++feature)
                    dot += weights[layer].W[w + feature] * aggregated[msg + feature];
                h_next[out + output] = relu(dot);
            }
        }
        h_current.swap(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << "/" << num_layers << " completato ("
                  << (num_edges + batch_size - 1) / batch_size << " batch di messaggi).\n";
    }

    std::cout << "Calcolo Softmax...\n";
#pragma omp parallel for schedule(static)
    for (int vertex = 0; vertex < num_nodes; ++vertex) {
        const size_t offset = static_cast<size_t>(vertex) * num_classes;
        float max_logit = h_current[offset];
        for (int c = 1; c < num_classes; ++c)
            max_logit = std::max(max_logit, h_current[offset + c]);
        float sum = 0.0f;
        for (int c = 0; c < num_classes; ++c) {
            h_current[offset + c] = std::exp(h_current[offset + c] - max_logit);
            sum += h_current[offset + c];
        }
        for (int c = 0; c < num_classes; ++c) h_current[offset + c] /= sum;
    }
    std::cout << "Elaborazione conclusa con successo!\n";
    return 0;
}
