#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "inference.h"

LayerWeights::LayerWeights(int in, int out)
    : in_dim(in), out_dim(out), W(static_cast<size_t>(in) * out, 0.0f) {}

namespace {
bool readInteger(const std::unordered_map<std::string, std::string>& metadata,
                 const std::string& key, int& value, std::string& error) {
    const auto entry = metadata.find(key);
    if (entry == metadata.end()) {
        error = "campo mancante nel manifest: " + key;
        return false;
    }
    try {
        size_t consumed = 0;
        value = std::stoi(entry->second, &consumed);
        if (consumed != entry->second.size()) throw std::invalid_argument("suffix");
    } catch (const std::exception&) {
        error = "valore non valido per " + key + ": " + entry->second;
        return false;
    }
    return true;
}
}

bool loadModelWeights(const std::string& folder_path,
                      const std::string& expected_dataset,
                      const std::vector<int>& expected_dimensions,
                      std::vector<LayerWeights>& layers,
                      std::string& error_message) {
    const std::string manifest_path = folder_path + "/metadata.txt";
    std::ifstream manifest(manifest_path);
    if (!manifest) {
        error_message = "impossibile aprire " + manifest_path;
        return false;
    }

    std::unordered_map<std::string, std::string> metadata;
    std::string line;
    while (std::getline(manifest, line)) {
        const size_t separator = line.find(':');
        if (separator != std::string::npos)
            metadata[line.substr(0, separator)] = line.substr(separator + 1);
    }

    const auto dataset = metadata.find("dataset");
    if (dataset == metadata.end() || dataset->second != expected_dataset) {
        error_message = "dataset dei pesi diverso da quello richiesto (" +
                        expected_dataset + ")";
        return false;
    }

    int format_version = 0;
    int num_layers = 0;
    if (!readInteger(metadata, "format_version", format_version, error_message) ||
        !readInteger(metadata, "num_layers", num_layers, error_message)) return false;
    if (format_version != 1) {
        error_message = "versione del formato pesi non supportata";
        return false;
    }
    if (expected_dimensions.size() != static_cast<size_t>(num_layers + 1)) {
        error_message = "numero di layer incompatibile";
        return false;
    }

    std::vector<LayerWeights> loaded;
    loaded.reserve(num_layers);
    for (int layer = 0; layer < num_layers; ++layer) {
        const std::string prefix = "layer_" + std::to_string(layer);
        int stored_in = 0;
        int stored_out = 0;
        if (!readInteger(metadata, prefix + "_in", stored_in, error_message) ||
            !readInteger(metadata, prefix + "_out", stored_out, error_message)) return false;
        if (stored_in != expected_dimensions[layer] ||
            stored_out != expected_dimensions[layer + 1]) {
            error_message = "dimensioni incompatibili al layer " + std::to_string(layer) +
                            ": file=" + std::to_string(stored_in) + "x" +
                            std::to_string(stored_out) + ", richieste=" +
                            std::to_string(expected_dimensions[layer]) + "x" +
                            std::to_string(expected_dimensions[layer + 1]);
            return false;
        }

        LayerWeights weights(stored_in, stored_out);
        const std::string path = folder_path + "/" + prefix + ".txt";
        std::ifstream values(path);
        if (!values) {
            error_message = "impossibile aprire " + path;
            return false;
        }
        size_t count = 0;
        float value = 0.0f;
        while (values >> value) {
            if (count == weights.W.size()) {
                error_message = "troppi valori in " + path;
                return false;
            }
            weights.W[count++] = value;
        }
        if (!values.eof()) {
            error_message = "valore numerico non valido in " + path;
            return false;
        }
        if (count != weights.W.size()) {
            error_message = "numero di valori errato in " + path + ": attesi " +
                            std::to_string(weights.W.size()) + ", letti " +
                            std::to_string(count);
            return false;
        }
        loaded.push_back(std::move(weights));
    }

    layers = std::move(loaded);
    return true;
}

float relu(float x) {
    return std::max(0.0f, x);
}
