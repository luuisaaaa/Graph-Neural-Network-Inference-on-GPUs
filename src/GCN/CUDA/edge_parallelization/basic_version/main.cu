#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include "../../../utilities/benchmark.h"
#include "../../../utilities/graph.h"
#include "../../../utilities/inference.h"

// Funzione per la gestione degli errori CUDA
void checkCudaError(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "Errore CUDA: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

// Kernel 1. Aggregazione: parte relativa alla somma dei vettori m (un thread per arco)
__global__ void aggregate_kernel(const int* edge_src, const int* edge_dest, const float* h_current, float* m_all, int num_edges, int feature_dim) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (e < num_edges) {
        int u = edge_src[e];
        int v = edge_dest[e];
        for (int f = 0; f < feature_dim; f++) {
            //Accesso non coalescente
            atomicAdd(&m_all[v * feature_dim + f], h_current[u * feature_dim + f]);
        }
    }
}

// Kernel 2. Calcolo della media per i vettori m, esecuzione fase di update e ReLU (un thread per nodo)
__global__ void update_kernel(const float* h_current, float* m_all, const float* W, float* h_next, const int* in_degree, int num_nodes, int current_dim, int next_dim) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < num_nodes) {
        int deg = in_degree[v];
        
        //Calcolo della media
        for (int j = 0; j < current_dim; j++) {
            //Accesso non coalescente
            m_all[v * current_dim + j] = (deg > 0) ? (m_all[v * current_dim + j] / (float)deg) : h_current[v * current_dim + j];
        }
        
        //Fase di update
        for (int i = 0; i < next_dim; i++) {
            float dot_product = 0.0f;
            for (int j = 0; j < current_dim; j++) {
                //Accesso condiviso: W
                //Accesso non coalescente: m_all
                dot_product += W[i * current_dim + j] * m_all[v * current_dim + j];
            }
            //ReLU
            //Accesso non coalescente
            h_next[v * next_dim + i] = fmax(0.0f, dot_product);
        }
    }
}

// Kernel 3. Softmax (un thread per nodo)
__global__ void softmax_kernel(float* h_current, int num_nodes, int num_classes) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (v < num_nodes) {
        //Accesso non coalescente
        float max_logit = h_current[v * num_classes + 0];
        
        //Ricerca del massimo valore nel vettore h
        for (int c = 1; c < num_classes; c++) {
            //Accesso non coalescente
            if (h_current[v * num_classes + c] > max_logit) {
                max_logit = h_current[v * num_classes + c];
            }
        }

        //Calcolo somma esponenziale
        float sum_exp = 0.0f;
        for (int c = 0; c < num_classes; c++) {
            //Accesso non coalescente
            h_current[v * num_classes + c] = expf(h_current[v * num_classes + c] - max_logit);
            sum_exp += h_current[v * num_classes + c];
        }

        //Calcolo vettore h
        for (int c = 0; c < num_classes; c++) {
            //Accesso non coalescente
            h_current[v * num_classes + c] /= sum_exp;
        }
    }
}


int main(int argc, char* argv[]){
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
        std::cerr << "Errore: i parametri dimensionali devono essere positivi." << std::endl;
        return 1;
    }

    // Controllo della disponibilità di una GPU CUDA
    int deviceCount = 0;
    checkCudaError(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        std::cerr << "Errore: Nessun dispositivo GPU abilitato CUDA rilevato." << std::endl;
        return 1;
    }
    int device;
    checkCudaError(cudaGetDevice(&device));

    std::string folderPath = "../../../../../dataset/converted/" + dataset_name;
    Graph g;
    int num_edges, num_nodes, feature_dim;

    std::cout << "========================================" << std::endl;
    std::cout << "Dataset: " << dataset_name << std::endl;
    std::cout << "Layer (L): " << num_layers << " | Hidden Dim: " << hidden_dim << " | Classi: " << num_classes << std::endl;
    std::cout << "Modalità: CUDA Edge-Parallel (1 Thread per Arco)" << std::endl;
    std::cout << "Dispositivo GPU CUDA in uso: " << device << std::endl;
    std::cout << "Percorso Grafo: " << folderPath << std::endl;
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

    // Calcolo della dimensione totale e salvataggio degli offset di partenza per W
    int total_weights_size = 0;
    std::vector<int> weight_offsets(num_layers, 0);
    for (int l = 0; l < num_layers; l++) {
        weight_offsets[l] = total_weights_size;
        total_weights_size += W[l].W.size();
    }

    // Copia di tutti i pesi in un unico array host
    std::vector<float> h_W_all(total_weights_size);
    for (int l = 0; l < num_layers; l++) {
        std::copy(W[l].W.begin(), W[l].W.end(), h_W_all.begin() + weight_offsets[l]);
    }

    // Preparazione dei dati host
    std::vector<int> h_edge_src = g.getEdgeSrc();
    const std::vector<int>& h_edge_dest = g.getEdgeDest();
    std::vector<int> h_in_degree = g.getInDegree();
    
    // Inizializzazione lineare delle feature dei nodi per formare un array 1D
    std::vector<float> h_h_current(num_nodes * feature_dim, 0.0f);
    for (int v = 0; v < num_nodes; ++v) {
        for (int f = 0; f < feature_dim; ++f) {
            h_h_current[v * feature_dim + f] = g.getVertex(v).features[f];
        }
    }

    // Allocazione memoria device
    int *d_edge_src, *d_edge_dest, *d_in_degree;
    float *d_h_current, *d_h_next, *d_m_all, *d_W_all;

    checkCudaError(cudaMalloc(&d_edge_src, num_edges * sizeof(int)));
    checkCudaError(cudaMalloc(&d_edge_dest, num_edges * sizeof(int)));
    checkCudaError(cudaMalloc(&d_in_degree, num_nodes * sizeof(int)));
    
    // Vettori per h_current, h_next e m_all
    int max_dim = std::max({feature_dim, hidden_dim, num_classes});
    checkCudaError(cudaMalloc(&d_h_next, num_nodes * max_dim * sizeof(float)));
    checkCudaError(cudaMalloc(&d_m_all, num_nodes * max_dim * sizeof(float)));
    checkCudaError(cudaMalloc(&d_h_current, num_nodes * max_dim * sizeof(float)));
    
    // Allocazione per tutti i pesi
    checkCudaError(cudaMalloc(&d_W_all, total_weights_size * sizeof(float)));

    // Trasferimento topologia grafo, feature Iniziali e pesi su GPU
    checkCudaError(cudaMemcpy(d_edge_src, h_edge_src.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_edge_dest, h_edge_dest.data(), num_edges * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_in_degree, h_in_degree.data(), num_nodes * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_h_current, h_h_current.data(), num_nodes * feature_dim * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_W_all, h_W_all.data(), total_weights_size * sizeof(float), cudaMemcpyHostToDevice));

    // Creazione dei CUDA events
    cudaEvent_t inference_begin, inference_end;
    checkCudaError(cudaEventCreate(&inference_begin));
    checkCudaError(cudaEventCreate(&inference_end));

    std::cout << "Inizio elaborazione inference CUDA..." << std::endl;

    // Inserimento dell'evento inference_begin 
    checkCudaError(cudaEventRecord(inference_begin));
    
    int current_dim = feature_dim;
    for (int l = 0; l < W.size(); l++) {
        int next_dim = W[l].out_dim;
        float* d_W_current_layer = d_W_all + weight_offsets[l];

        // Inizializzazione a zero del vettore m_all
        checkCudaError(cudaMemset(d_m_all, 0, num_nodes * current_dim * sizeof(float)));

        // Configurazione delle griglie
        int threadsPerBlock = 256;
        int blocksPerGridEdges = (num_edges + threadsPerBlock - 1) / threadsPerBlock;
        int blocksPerGridNodes = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;

        // Aggregazione
        aggregate_kernel<<<blocksPerGridEdges, threadsPerBlock>>>(d_edge_src, d_edge_dest, d_h_current, d_m_all, num_edges, current_dim);
        checkCudaError(cudaGetLastError());

        // Update
        update_kernel<<<blocksPerGridNodes, threadsPerBlock>>>(d_h_current, d_m_all, d_W_current_layer, d_h_next, d_in_degree, num_nodes, current_dim, next_dim);
        checkCudaError(cudaGetLastError());

        // Scambio dei puntatori
        float* temp = d_h_current;
        d_h_current = d_h_next;
        d_h_next = temp;

        current_dim = next_dim;

        std::cout << "-> Livello " << l + 1 << "/" << num_layers << " completato." << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;
    
    // Esecuzione Softmax
    int threadsPerBlock = 256;
    int blocksPerGridNodes = (num_nodes + threadsPerBlock - 1) / threadsPerBlock;
    softmax_kernel<<<blocksPerGridNodes, threadsPerBlock>>>(d_h_current, num_nodes, num_classes);
    checkCudaError(cudaGetLastError());

    // Inserimento dell'evento inference_end 
    checkCudaError(cudaEventRecord(inference_end));
    
    // Attesa che l'evento inference_end venga gestito
    checkCudaError(cudaEventSynchronize(inference_end));

    // Calcolo tempo di esecuzione
    float inference_ms = 0.0f;
    checkCudaError(cudaEventElapsedTime(&inference_ms, inference_begin, inference_end));

    // Copia dell'output sul vettore host finale
    std::vector<float> h_output(num_nodes * num_classes, 0.0f);
    checkCudaError(cudaMemcpy(h_output.data(), d_h_current, num_nodes * num_classes * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Elaborazione conclusa con successo!" << std::endl;

    // Stampa standardizzata del throughput e del tempo
    reportResults("cuda-edge-parallel-basic", num_nodes, num_edges, num_classes, num_layers, inference_ms);

    // Pulizia finale della memoria GPU e degli eventi
    checkCudaError(cudaEventDestroy(inference_begin));
    checkCudaError(cudaEventDestroy(inference_end));
    checkCudaError(cudaFree(d_edge_src));
    checkCudaError(cudaFree(d_edge_dest));
    checkCudaError(cudaFree(d_in_degree));
    checkCudaError(cudaFree(d_h_current));
    checkCudaError(cudaFree(d_h_next));
    checkCudaError(cudaFree(d_m_all));
    checkCudaError(cudaFree(d_W_all));

    return 0;
}