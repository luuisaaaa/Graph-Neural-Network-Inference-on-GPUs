#ifndef GRAPH_H
#define GRAPH_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

struct VertexData {
    std::vector<float> features;
    int label = -1;
    std::vector<int> neighbors;
};

class Graph {
public:
    Graph() = default;
    ~Graph() = default;

    bool loadGraph(const std::string& folderPath);
    std::vector<int> getNeighbors(int v) const;
    VertexData getVertex(int v) const;

    int getNumNodes() const;
    int getNumEdges() const;
    int getFeatureDim() const;

    const std::vector<int>& getRowPointers() const;
    const std::vector<int>& getColumnIndices() const;
    const std::vector<float>& getNodeFeatures() const;
    const std::vector<int>& getLabels() const;

private:
    int num_nodes = 0;
    int num_edges = 0;
    int feature_dim = 0;

    std::vector<int> row_pointers;
    std::vector<int> column_indices;
    std::vector<float> node_features;
    std::vector<int> labels;

    bool loadRowPointers(const std::string& filePath, std::vector<int>& vec);
    bool loadColumnIndices(const std::string& filePath, std::vector<int>& vec);
    bool loadMetadata(const std::string& filePath);
    bool loadFeatures(const std::string& filePath, std::vector<float>& vec);
    bool loadLabels(const std::string& filePath, std::vector<int>& vec);
};

#endif
