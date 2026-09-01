#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include "../utilities/benchmark.h"
#include "../utilities/graph.h"
#include "../utilities/inference.h"

int main(int argc, char* argv[])
{
    //Lettura dei parametri da terminale
    if (argc < 5) {
        std::cerr << "Errore: Parametri mancanti." << std::endl;
        std::cerr << "Uso: " << argv[0] << " <nome_dataset> <hidden_dim> <num_classes> <num_layers>" << std::endl;
        return 1;
    }

    std::string dataset_name = argv[1];
    int hidden_dim = std::stoi(argv[2]);
    int num_classes = std::stoi(argv[3]);
    int num_layers = std::stoi(argv[4]);

    if (num_layers < 1) {
        std::cerr << "Errore: Il numero di layer deve essere almeno 1." << std::endl;
        return 1;
    }

    std::string folderPath = "../../../dataset/converted/" + dataset_name;
    Graph g;
    std::vector<std::vector<float>> h_current;
    int num_edges, num_nodes, feature_dim;

    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim << " | Classi: " << num_classes << std::endl;
    std::cout << "Percorso: " << folderPath << std::endl;
    std::cout << "========================================" << std::endl;

    // Caricamento grafo
    if (!g.loadGraph(folderPath)) {
        std::cerr << "Errore nel caricamento del grafo!" << std::endl;
        return 1;
    }
    
    num_edges = g.getNumEdges();
    num_nodes = g.getNumNodes();
    feature_dim = g.getFeatureDim();

    //Creazione dinamica dei Pesi (W) in base al numero di layer
    std::vector<LayerWeights> W;
    
    if (num_layers == 1) {
        W.push_back(LayerWeights(feature_dim, num_classes));
    } 
    else {
        W.push_back(LayerWeights(feature_dim, hidden_dim));
    
        for (int l = 1; l < num_layers - 1; ++l) {
            W.push_back(LayerWeights(hidden_dim, hidden_dim));
        }

        W.push_back(LayerWeights(hidden_dim, num_classes));
    }

    //Inizializzazione h^(0) = x_v per tutti i nodi
    h_current.resize(num_nodes);
    for (int v = 0; v < num_nodes; ++v) {
        h_current[v] = g.getVertex(v).features;
    }

    std::cout << "Inizio elaborazione inference..." << std::endl;
    const auto inference_begin = BenchmarkClock::now();

    //Iterazione su tutti i layer L della rete
    int current_dim = feature_dim;
    for (int l = 0; l < W.size(); l++) {
        int next_dim = W[l].out_dim;

        // Struttura per h^(l+1)
        std::vector<std::vector<float>> h_next(num_nodes, std::vector<float>(next_dim, 0.0f));

        // Vettore di supporto per l'aggregazione di un singolo nodo
        std::vector<float> m_v(current_dim, 0.0f);

        // Esecuzione iterativa su ogni nodo v del grafo
        for (int v = 0; v < num_nodes; v++) {
            VertexData v_data = g.getVertex(v);
            const std::vector<int> &neighbors = v_data.neighbors;

            // Inizializza m_v a zero per il nodo corrente
            std::fill(m_v.begin(), m_v.end(), 0.0f);

            // Aggregazione (media)
            if (!neighbors.empty()) {
                int degree = neighbors.size();

                // Somma delle feature di tutti i vicini
                for (int k = 0; k < degree; k++) {
                    int u = neighbors[k];
                    for (int f = 0; f < current_dim; f++) {
                        m_v[f] += h_current[u][f];
                    }
                }

                // Calcolo della media
                for (int f = 0; f < current_dim; f++) {
                    m_v[f] /= static_cast<float>(degree);
                }
            } 
            else {
                m_v = h_current[v];
            }

            // Update
            for (int i = 0; i < next_dim; i++) {
                float dot_product = 0.0f;
                for (int j = 0; j < current_dim; j++) {
                    dot_product += W[l].W[i * current_dim + j] * m_v[j];
                }
                
                // Funzione di attivazione ReLU
                h_next[v][i] = relu(dot_product);
            }
        }

        // Passaggio dell'output come input al layer successivo
        h_current = std::move(h_next);
        current_dim = next_dim;
        std::cout << "-> Livello " << l + 1 << "/" << num_layers << " completato." << std::endl;

    }

    std::cout << "Calcolo Softmax..." << std::endl;

    //Softmax
    for (int v = 0; v < num_nodes; v++) {
        int num_classes_out = h_current[v].size();

        // Trova il valore massimo per la stabilità numerica
        float max_logit = h_current[v][0];
        for (int c = 1; c < num_classes_out; c++) {
            if (h_current[v][c] > max_logit) {
                max_logit = h_current[v][c];
            }
        }

        // Calcola l'esponenziale per ogni logit e la somma totale
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes_out; c++) {
            h_current[v][c] = std::exp(h_current[v][c] - max_logit);
            sum_exp += h_current[v][c];
        }

        // Normalizzazione
        for (int c = 0; c < num_classes_out; c++) {
            h_current[v][c] /= sum_exp;
        }
    }

    const auto inference_end = BenchmarkClock::now();
    std::vector<float> output(static_cast<size_t>(num_nodes) * num_classes);
    for (int v = 0; v < num_nodes; ++v) {
        std::copy(h_current[v].begin(), h_current[v].end(),
                  output.begin() + static_cast<size_t>(v) * num_classes);
    }
    reportResults("sequential", output, g.getLabels(), num_nodes, num_edges, num_classes,
                  num_layers, elapsedMilliseconds(inference_begin, inference_end));

    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}
