#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include "../utilities/benchmark.h"
#include <vector>
#include "../utilities/graph.h"
#include "../utilities/inference.h"

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Uso: " << argv[0]
                  << " <nome_dataset> <hidden_dim> <num_classes> <num_layers> <cartella_pesi>\n";
        return 1;
    }

    const std::string dataset_name = argv[1];
    const int hidden_dim = std::stoi(argv[2]);
    const int num_classes = std::stoi(argv[3]);
    const int num_layers = std::stoi(argv[4]);
    const std::string weights_path = argv[5];
    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1) {
        std::cerr << "Errore: hidden_dim, num_classes e num_layers devono essere positivi.\n";
        return 1;
    }

    const std::string folder_path = "../../../dataset/converted/" + dataset_name;
    Graph graph;
    std::cout << "========================================\n"
              << "Dataset: " << dataset_name << '\n'
              << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim
              << " | Classi: " << num_classes << '\n'
              << "Direzione CSR: source -> destination\n"
              << "Percorso: " << folder_path << '\n'
              << "========================================\n";
    if (!graph.loadGraph(folder_path)) {
        std::cerr << "Errore nel caricamento del grafo!\n";
        return 1;
    }

    const int num_nodes = graph.getNumNodes();
    const int num_edges = graph.getNumEdges();
    const int feature_dim = graph.getFeatureDim();
    const std::vector<int>& row_pointers = graph.getRowPointers();
    const std::vector<int>& destinations = graph.getColumnIndices();

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

    std::vector<int> in_degree(num_nodes, 0);
    for (int destination : destinations) ++in_degree[destination];

    std::vector<float> h_current = graph.getNodeFeatures();
    int current_dim = feature_dim;
    std::cout << "Inizio elaborazione inference...\n";
    const auto inference_begin = BenchmarkClock::now();

    for (int layer = 0; layer < num_layers; ++layer) {
        const int next_dim = weights[layer].out_dim;
        std::vector<float> aggregated(static_cast<size_t>(num_nodes) * current_dim, 0.0f);
        std::vector<float> h_next(static_cast<size_t>(num_nodes) * next_dim, 0.0f);

        // Ogni riga CSR appartiene alla sorgente; gli indici nella riga sono
        // destinazioni. Il nodo destinazione aggrega quindi messaggi entranti.
        for (int source = 0; source < num_nodes; ++source) {
            const size_t source_offset = static_cast<size_t>(source) * current_dim;
            for (int edge = row_pointers[source]; edge < row_pointers[source + 1]; ++edge) {
                const int destination = destinations[edge];
                const size_t destination_offset = static_cast<size_t>(destination) * current_dim;
                for (int feature = 0; feature < current_dim; ++feature) {
                    aggregated[destination_offset + feature] +=
                        h_current[source_offset + feature];
                }
            }
        }

        for (int vertex = 0; vertex < num_nodes; ++vertex) {
            const size_t input_offset = static_cast<size_t>(vertex) * current_dim;
            const size_t output_offset = static_cast<size_t>(vertex) * next_dim;
            for (int output = 0; output < next_dim; ++output) {
                float dot_product = 0.0f;
                const size_t weight_offset = static_cast<size_t>(output) * current_dim;
                for (int feature = 0; feature < current_dim; ++feature) {
                    const float message = in_degree[vertex] > 0
                        ? aggregated[input_offset + feature] / static_cast<float>(in_degree[vertex])
                        : h_current[input_offset + feature];
                    dot_product += weights[layer].W[weight_offset + feature] * message;
                }
                h_next[output_offset + output] = relu(dot_product);
            }
        }

        h_current.swap(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << layer + 1 << '/' << num_layers << " completato.\n";
    }

    std::cout << "Calcolo Softmax...\n";
    for (int vertex = 0; vertex < num_nodes; ++vertex) {
        const size_t offset = static_cast<size_t>(vertex) * num_classes;
        float maximum = h_current[offset];
        for (int class_id = 1; class_id < num_classes; ++class_id)
            maximum = std::max(maximum, h_current[offset + class_id]);
        float sum = 0.0f;
        for (int class_id = 0; class_id < num_classes; ++class_id) {
            h_current[offset + class_id] = std::exp(h_current[offset + class_id] - maximum);
            sum += h_current[offset + class_id];
        }
        for (int class_id = 0; class_id < num_classes; ++class_id)
            h_current[offset + class_id] /= sum;
    }

    const auto inference_end = BenchmarkClock::now();
    std::uint64_t weight_elements = 0;
    for (const LayerWeights& layer_weights : weights)
        weight_elements += layer_weights.W.size();
    const int max_dim = std::max({feature_dim, hidden_dim, num_classes});
    const std::uint64_t working_bytes =
        3ULL * num_nodes * max_dim * sizeof(float) +
        static_cast<std::uint64_t>(num_nodes) * sizeof(int);
    const MemoryMetrics memory = makeMemoryMetrics(
        num_nodes, num_edges, feature_dim, weight_elements, working_bytes);
    reportResults("sequential", h_current, graph.getLabels(), num_nodes, num_edges, num_classes,
                  num_layers, elapsedMilliseconds(inference_begin, inference_end), memory);

    std::cout << "Elaborazione conclusa con successo!\n";
    return 0;
}
