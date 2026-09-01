#include <vector>
#include <algorithm>
#include <limits>
#include <random>
#include "inference.h"

//Genera in modo casuale la matrice W^l
LayerWeights::LayerWeights(int in, int out) {
    this->in_dim = in;
    this->out_dim = out;

    int total_elements = in * out;
    this->W.resize(total_elements);

    // Seed fisso e generatore standard: eseguibili diversi ricevono gli stessi
    // pesi, requisito necessario per confrontare gli output numerici.
    static std::mt19937 generator(42);
    for (int i = 0; i < total_elements; ++i) {
        this->W[i] = static_cast<float>(generator()) /
                     static_cast<float>(std::numeric_limits<std::mt19937::result_type>::max());
    }
}

// Funzione di attivazione ReLU
float relu(float x) {
    return std::max(0.0f, x);
}
