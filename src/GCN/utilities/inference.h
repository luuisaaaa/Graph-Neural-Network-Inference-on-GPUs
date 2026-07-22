#ifndef INFERENCE_H
#define INFERENCE_H

#include <vector>

// Struttura che per pesi W^(l)
struct LayerWeights {
    int in_dim;
    int out_dim;
    std::vector<float> W;
    
    // Costruttore con inizializzazione fittizia per i pesi
    LayerWeights(int in, int out);
};


float relu(float x);

#endif