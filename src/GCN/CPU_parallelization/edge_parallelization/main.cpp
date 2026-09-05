#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <omp.h>
#include "../../utilities/graph.h"
#include "../../utilities/inference.h"
#include "../../utilities/benchmark.h"

int main(int argc, char* argv[])
{
    if (argc < 6) {
        std::cerr << "Errore: Parametri mancanti." << std::endl;
        std::cerr << "Uso: " << argv[0] << " <nome_dataset> <hidden_dim> <num_classes> <num_layers> <cartella_pesi>" << std::endl;
        return 1;
    }

    std::string dataset_name = argv[1];
    int hidden_dim = std::stoi(argv[2]);
    int num_classes = std::stoi(argv[3]);
    int num_layers = std::stoi(argv[4]);
    std::string weights_path = argv[5];

    if (hidden_dim < 1 || num_classes < 1 || num_layers < 1) {
        std::cerr << "Errore: hidden_dim, num_classes e num_layers devono essere positivi.\n" << std::endl;
        return 1;
    }

    std::string folderPath = "../../../../dataset/converted/" + dataset_name;
    Graph g;
    int num_edges, num_nodes, feature_dim;

    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim << " | Classi: " << num_classes << std::endl;
    std::cout << "Modalità: Edge-Parallel" << std::endl;
    std::cout << "Thread OpenMP massimi: " << omp_get_max_threads() << std::endl;
    std::cout << "Percorso: " << folderPath << std::endl;
    std::cout << "========================================" << std::endl;

    if (!g.loadGraph(folderPath)) {
        std::cerr << "Errore nel caricamento del grafo!" << std::endl;
        return 1;
    }
    
    num_edges = g.getNumEdges();
    num_nodes = g.getNumNodes();
    feature_dim = g.getFeatureDim();

    // Creazione delle dimensioni attese per i layer
    std::vector<int> expected_dimensions;
    expected_dimensions.push_back(feature_dim);
    for (int l = 1; l < num_layers; ++l) {
        expected_dimensions.push_back(hidden_dim);
    }
    expected_dimensions.push_back(num_classes);

    // Caricamento dei Pesi (W) da file
    std::vector<LayerWeights> W;
    std::string error_message;
    if (!loadModelWeights(weights_path, dataset_name, expected_dimensions, W, error_message)) {
        std::cerr << "Errore nel caricamento dei pesi: " << error_message << std::endl;
        return 1;
    }
    std::cout << "Pesi precaricati da: " << weights_path << std::endl;

    // Inizializzazione h^(0) = x_v per tutti i nodi
    std::vector<float> h_current = g.getNodeFeatures();

    // Creazione delle strutture dati necessarie
    std::vector<int> edge_src = g.getEdgeSrc();                     //archi: sorgente
    const std::vector<int>& edge_dest = g.getEdgeDest();            //archi: destinazione
    std::vector<int> in_degree = g.getInDegree();                   //numero di archi per nodi

    // Iterazione su tutti i layer L della rete
    std::cout << "Inizio elaborazione inference..." << std::endl;
    const auto inference_begin = BenchmarkClock::now();
    int current_dim = feature_dim;
    for (int l = 0; l < W.size(); l++) {
        int next_dim = W[l].out_dim;

        std::vector<float> h_next(num_nodes * next_dim, 0.0f);
        std::vector<float> m_all(num_nodes * current_dim, 0.0f);

        // Aggregazione (media)
        // Calcolo somma per vettori m
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < num_edges; e++) {
            int u = edge_src[e];
            int v = edge_dest[e];
            
            for (int f = 0; f < current_dim; f++) {
                #pragma omp atomic
                m_all[v * current_dim + f] += h_current[u * current_dim + f];
            }
        }

        // Calcolo vettori m + fase di update
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < num_nodes; v++) {
            if (in_degree[v] > 0) {
                for (int f = 0; f < current_dim; f++) {
                    m_all[v * current_dim + f] /= static_cast<float>(in_degree[v]);
                }
            } else {
                for (int f = 0; f < current_dim; f++) {
                    m_all[v * current_dim + f] = h_current[v * current_dim + f];
                }
            }

            for (int i = 0; i < next_dim; i++) {
                float dot_product = 0.0f;
                for (int j = 0; j < current_dim; j++) {
                    dot_product += W[l].W[i * current_dim + j] * m_all[v * current_dim + j];
                }
                h_next[v * next_dim + i] = relu(dot_product);
            }
        }

        h_current = std::move(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << l + 1 << "/" << num_layers << " completato." << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;

    // Softmax parallelizzato sui nodi
    #pragma omp parallel for schedule(static)
    for (int v = 0; v < num_nodes; v++) {
        int num_classes_out = num_classes;
        float max_logit = h_current[v * num_classes_out + 0];
        for (int c = 1; c < num_classes_out; c++) {
            if (h_current[v * num_classes_out + c] > max_logit) {
                max_logit = h_current[v * num_classes_out + c];
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes_out; c++) {
            h_current[v * num_classes_out + c] = std::exp(h_current[v * num_classes_out + c] - max_logit);
            sum_exp += h_current[v * num_classes_out + c];
        }

        for (int c = 0; c < num_classes_out; c++) {
            h_current[v * num_classes_out + c] /= sum_exp;
        }
    }
    const auto inference_end = BenchmarkClock::now();

    std::cout << "Elaborazione conclusa con successo!" << std::endl;

    std::uint64_t weight_elements = 0;
    for (const LayerWeights& layer_weights : W) {
        weight_elements += layer_weights.W.size();
    }
    
    const int max_dim = std::max({feature_dim, hidden_dim, num_classes});
    const std::uint64_t copied_structs_bytes = (static_cast<std::uint64_t>(num_edges) + num_nodes) * sizeof(int);
    const std::uint64_t activation_bytes = 2ULL * num_nodes * max_dim * sizeof(float);
    
    const MemoryMetrics memory = makeMemoryMetrics(
        num_nodes, num_edges, feature_dim, weight_elements, 
        copied_structs_bytes + activation_bytes
    );

    reportResults("cpu-edge-parallel", h_current, g.getLabels(), num_nodes, num_edges, num_classes, num_layers,
                  elapsedMilliseconds(inference_begin, inference_end), memory);

    return 0;
}