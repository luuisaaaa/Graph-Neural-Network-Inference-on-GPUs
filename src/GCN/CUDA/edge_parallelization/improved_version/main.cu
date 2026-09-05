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

// Kernel 1. Aggregazione: parte relativa alla somma dei vettori m (un warp per arco)
__global__ void aggregate_kernel(const int* edge_src, const int* edge_dest, const float* h_current, float* m_all, int num_edges, int feature_dim) {
    int e = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (e < num_edges) {
        int u = edge_src[e];
        int v = edge_dest[e];
        for (int f = threadIdx.x; f < feature_dim; f += blockDim.x) {
            // Accesso coalescente
            atomicAdd(&m_all[v * feature_dim + f], h_current[u * feature_dim + f]);
        }
    }
}

// Kernel 2. Calcolo della media per i vettori m, esecuzione fase di update e ReLU (un warp per nodo)
__global__ void update_kernel(const float* h_current, float* m_all, const float* W, float* h_next, const int* in_degree, int num_nodes, int current_dim, int next_dim) {
    int v = blockIdx.x * blockDim.y + threadIdx.y;
    
    if (v < num_nodes) {
        int deg = in_degree[v];
        
        //Calcolo della media
        for (int j = threadIdx.x; j < current_dim; j += blockDim.x) {
            m_all[v * current_dim + j] = (deg > 0) ? (m_all[v * current_dim + j] / (float)deg) : h_current[v * current_dim + j];
        }

        __syncwarp();

        //Fase di update
        for (int i = threadIdx.x; i < next_dim; i += blockDim.x) {
            float dot_product = 0.0f;
            for (int j = 0; j < current_dim; j++) {
                //Accesso coalescente: W
                //Accesso condiviso: m_all
                dot_product += W[j * next_dim + i] * m_all[v * current_dim + j];
            }
            //ReLU
            //Accesso coalescente
            h_next[v * next_dim + i] = fmax(0.0f, dot_product);
        }
    }
}

// Kernel 3. Softmax (un warp per nodo)
__global__ void softmax_kernel(float* h_current, int num_nodes, int num_classes) {
    int v = blockIdx.x * blockDim.y + threadIdx.y;
    __shared__ float s_data[8][32];

    float max_logit = -INFINITY;
    
    if (v < num_nodes) {
        //Ricerca del massimo valore nel vettore h
        for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
            if (h_current[v * num_classes + c] > max_logit) {
                max_logit = h_current[v * num_classes + c];
            }
        }

        s_data[threadIdx.y][threadIdx.x] = max_logit;
        __syncwarp();

        //Thread 0 di ogni warp definisce il minimo
        if (threadIdx.x == 0) {
            float overall_max = s_data[threadIdx.y][0];
            for (int i = 1; i < 32; i++) {
                if (s_data[threadIdx.y][i] > overall_max) {
                    overall_max = s_data[threadIdx.y][i];
                }
            }
            s_data[threadIdx.y][0] = overall_max;
        }

        //Lettura del massimo da parte di tutti i thread del warp
        __syncwarp();
        max_logit = s_data[threadIdx.y][0];
        float sum_exp = 0.0f;

        //Calcolo somma esponenziale (divisa tra thread)
        for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
            h_current[v * num_classes + c] = expf(h_current[v * num_classes + c] - max_logit);
            sum_exp += h_current[v * num_classes + c];
        }
        s_data[threadIdx.y][threadIdx.x] = sum_exp;
        __syncwarp();

        //Thread 0 di ogni warp calcola la somma totale
        if (threadIdx.x == 0) {
            float overall_sum = 0.0f;
            for (int i = 0; i < 32; i++) {
                overall_sum += s_data[threadIdx.y][i];
            }
            s_data[threadIdx.y][0] = overall_sum;
        }

        //Lettura della somma esponenziale da parte di tutti i thread del warp
        __syncwarp();
        float total_sum_exp = s_data[threadIdx.y][0];

        //Calcolo vettore h
        for (int c = threadIdx.x; c < num_classes; c += blockDim.x) {
            h_current[v * num_classes + c] /= total_sum_exp;
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

    if (num_layers < 1) {
        std::cerr << "Errore: Il numero di layer deve essere almeno 1." << std::endl;
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
    std::cout << "Modalità: CUDA Edge-Parallel (1 warp per arco)" << std::endl;
    std::cout << "Dispositivo GPU CUDA in uso: " << device << std::endl;
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

    // Calcolo della dimensione totale e salvataggio degli offset di partenza per W
    int total_weights_size = 0;
    std::vector<int> weight_offsets(num_layers, 0);
    for (int l = 0; l < num_layers; l++) {
        weight_offsets[l] = total_weights_size;
        total_weights_size += W[l].W.size();
    }

    // Copia di tutti i pesi in un unico array host con trasposizione
    // La trasposizione serve per avere un accesso coalescente a W nel kernel 2
    std::vector<float> h_W_all(total_weights_size);
    for (int l = 0; l < num_layers; l++) {
        int in_d = W[l].in_dim;
        int out_d = W[l].out_dim;
        int offset = weight_offsets[l];
        
        for (int i = 0; i < out_d; i++) {
            for (int j = 0; j < in_d; j++) {
                // Indice originale (row-major) nel singolo layer
                int original_idx = i * in_d + j;
                
                // Indice trasposto (column-major) nel singolo layer
                int new_idx = j * out_d + i;
                
                // Scrittura nel vettore host globale sommando l'offset del layer
                h_W_all[offset + new_idx] = W[l].W[original_idx];
            }
        }
    }

    // Preparazione dei dati host
    std::vector<int> h_edge_src = g.getEdgeSrc();
    const std::vector<int>& h_edge_dest = g.getEdgeDest();
    std::vector<int> h_in_degree = g.getInDegree();
    const std::vector<float>& h_h_current = g.getNodeFeatures();

    // Allocazione memoria device
    int *d_edge_src, *d_edge_dest, *d_in_degree;
    float *d_h_current, *d_h_next, *d_m_all, *d_W_all;

    checkCudaError(cudaMalloc(&d_edge_src, num_edges * sizeof(int)));
    checkCudaError(cudaMalloc(&d_edge_dest, num_edges * sizeof(int)));
    checkCudaError(cudaMalloc(&d_in_degree, num_nodes * sizeof(int)));
    
    // Vettori per h_current, h_next e m_all
    // Si considera come dimensione il caso peggiore
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
        dim3 threadsPerBlock(32, 8);        //Blocchi che contengono 8 warps
        dim3 blocksPerGridEdges((num_edges + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);
        dim3 blocksPerGridNodes((num_nodes + threadsPerBlock.y - 1) / threadsPerBlock.y, 1);

        // Aggregazione
        aggregate_kernel<<<blocksPerGridEdges, threadsPerBlock>>>(d_edge_src, d_edge_dest, d_h_current, d_m_all, num_edges, current_dim);
        checkCudaError(cudaGetLastError());

        // Update
        update_kernel<<<blocksPerGridNodes, threadsPerBlock>>>(d_h_current, d_m_all, d_W_current_layer, d_h_next, d_in_degree, num_nodes, current_dim, next_dim);
        checkCudaError(cudaGetLastError());

        // Scambio dei puntatori per il layer successivo e aggiornamento della dimensione
        float* temp = d_h_current;
        d_h_current = d_h_next;
        d_h_next = temp;

        current_dim = next_dim;

        std::cout << "-> Livello " << l + 1 << "/" << num_layers << " completato." << std::endl;
    }

    std::cout << "Calcolo Softmax..." << std::endl;
    
    // Esecuzione Softmax
    dim3 threadsPerBlockSoftmax(32, 8);
    dim3 blocksPerGridNodesSoftmax((num_nodes + threadsPerBlockSoftmax.y - 1) / threadsPerBlockSoftmax.y, 1);
    softmax_kernel<<<blocksPerGridNodesSoftmax, threadsPerBlockSoftmax>>>(d_h_current, num_nodes, num_classes);
    checkCudaError(cudaGetLastError());
    
    // Inserimento dell'evento inference_begin 
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

    // Calcolo memorie e stampa standardizzata
    const std::uint64_t host_working_bytes = 
        (2ULL * num_edges + num_nodes) * sizeof(int) + 
        total_weights_size * sizeof(float) + 
        static_cast<std::uint64_t>(num_layers) * sizeof(int) + 
        static_cast<std::uint64_t>(num_nodes) * num_classes * sizeof(float);
        
    const std::uint64_t device_bytes = 
        (2ULL * num_edges + num_nodes) * sizeof(int) + 
        3ULL * num_nodes * max_dim * sizeof(float) + 
        total_weights_size * sizeof(float);
        
    const MemoryMetrics memory = makeMemoryMetrics(
        num_nodes, num_edges, feature_dim, total_weights_size, host_working_bytes, device_bytes);
        
    reportResults("cuda-edge-parallel-improved", h_output, g.getLabels(), num_nodes, num_edges, 
                  num_classes, num_layers, inference_ms, memory);

    checkCudaError(cudaEventDestroy(inference_begin));
    checkCudaError(cudaEventDestroy(inference_end));

    // Pulizia finale della memoria GPU
    checkCudaError(cudaFree(d_edge_src));
    checkCudaError(cudaFree(d_edge_dest));
    checkCudaError(cudaFree(d_in_degree));
    checkCudaError(cudaFree(d_h_current));
    checkCudaError(cudaFree(d_h_next));
    checkCudaError(cudaFree(d_m_all));
    checkCudaError(cudaFree(d_W_all));

    return 0;
}