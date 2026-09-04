#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <vector>

// Struttura che per pesi W^(l)
struct LayerWeights {
    int in_dim;
    int out_dim;
    std::vector<float> W;
    
    // Costruisce la matrice; i valori vengono poi caricati da file.
    LayerWeights(int in, int out);
};

bool loadModelWeights(const std::string& folder_path,
                      const std::string& expected_dataset,
                      const std::vector<int>& expected_dimensions,
                      std::vector<LayerWeights>& layers,
                      std::string& error_message);

float relu(float x);

#endif
