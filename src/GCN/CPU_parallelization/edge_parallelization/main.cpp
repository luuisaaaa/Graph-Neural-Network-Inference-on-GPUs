#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <omp.h>
#include "../../utilities/graph.h"
#include "../../utilities/inference.h"

int main(int argc, char* argv[])
{
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

    std::string folderPath = "../../../../dataset/converted/" + dataset_name;
    Graph g;
    std::vector<std::vector<float>> h_current;
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

    //Creazione dinamica dei Pesi (W) in base al numero di layer
    std::vector<LayerWeights> W;
    if (num_layers == 1) {
        W.push_back(LayerWeights(feature_dim, num_classes));
    } else {
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

    
    //Creazione delle strutture dati necessarie
    std::vector<int> edge_src = g.getEdgeSrc();                     //archi: sorgente
    const std::vector<int>& edge_dest = g.getEdgeDest();            //archi: destinazione
    std::vector<int> in_degree = g.getInDegree();                   //numero di archi per nodi
    

    //Iterazione su tutti i layer L della rete
    std::cout << "Inizio elaborazione inference..." << std::endl;
    int current_dim = feature_dim;
    for (int l = 0; l < W.size(); l++) {
        int next_dim = W[l].out_dim;

        std::vector<std::vector<float>> h_next(num_nodes, std::vector<float>(next_dim, 0.0f));
        std::vector<std::vector<float>> m_all(num_nodes, std::vector<float>(current_dim, 0.0f));

        // Aggregazione (media)
        // Calcolo somma per vettori m
        #pragma omp parallel for schedule(static)
        for (int e = 0; e < num_edges; e++) {
            int u = edge_src[e];
            int v = edge_dest[e];
            
            for (int f = 0; f < current_dim; f++) {
                #pragma omp atomic
                m_all[v][f] += h_current[u][f];
            }
        }

        // Calcolo vettori m + fase di update
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < num_nodes; v++) {
            if (in_degree[v] > 0) {
                for (int f = 0; f < current_dim; f++) {
                    m_all[v][f] /= static_cast<float>(in_degree[v]);
                }
            } else {
                for (int f = 0; f < current_dim; f++) {
                    m_all[v][f] = h_current[v][f];
                }
            }

            for (int i = 0; i < next_dim; i++) {
                float dot_product = 0.0f;
                for (int j = 0; j < current_dim; j++) {
                    dot_product += W[l].W[i * current_dim + j] * m_all[v][j];
                }
                h_next[v][i] = relu(dot_product);
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
        int num_classes_out = h_current[v].size();
        float max_logit = h_current[v][0];
        for (int c = 1; c < num_classes_out; c++) {
            if (h_current[v][c] > max_logit) {
                max_logit = h_current[v][c];
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes_out; c++) {
            h_current[v][c] = std::exp(h_current[v][c] - max_logit);
            sum_exp += h_current[v][c];
        }

        for (int c = 0; c < num_classes_out; c++) {
            h_current[v][c] /= sum_exp;
        }
    }

    std::cout << "Elaborazione conclusa con successo!" << std::endl;
    return 0;
}