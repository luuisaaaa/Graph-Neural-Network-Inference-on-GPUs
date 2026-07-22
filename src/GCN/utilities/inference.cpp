#include <vector>
#include <cstdlib>
#include <algorithm>
#include "inference.h"

//Genera in modo casuale la matrice W^l
LayerWeights::LayerWeights(int in, int out) {
    this->in_dim = in;
    this->out_dim = out;

    int total_elements = in * out;
    this->W.resize(total_elements);

    // Singolo ciclo for su tutti gli elementi contigui
    for (int i = 0; i < total_elements; ++i) {
        this->W[i] = (float)rand() / (float)RAND_MAX;
    }
}

// Funzione di attivazione ReLU
float relu(float x) {
    return std::max(0.0f, x);
}