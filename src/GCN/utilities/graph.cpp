#include "graph.h"
#include <algorithm>

// --- Getter ---
int Graph::getNumNodes() const {
    return num_nodes;
}

int Graph::getNumEdges() const {
    return num_edges;
}

int Graph::getFeatureDim() const {
    return feature_dim;
}

const std::vector<int>& Graph::getRowPointers() const {
    return row_pointers;
}

const std::vector<int>& Graph::getColumnIndices() const {
    return column_indices;
}

const std::vector<float>& Graph::getNodeFeatures() const {
    return node_features;
}

const std::vector<int>& Graph::getLabels() const {
    return labels;
}

// --- Implementazione delle funzioni di lettura dei file ---
//Lettura dei Metadati
bool Graph::loadMetadata(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire il file di metadata: " << filePath << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string key;
        int value;
        if (std::getline(ss, key, ':') && ss >> value) {
            if (key == "num_nodes") num_nodes = value;
            else if (key == "num_edges") num_edges = value;
            else if (key == "feature_dim") feature_dim = value;
        }
    }
    file.close();
    return true;
}

//Lettura dei Row Pointers (Interi, uno per riga)
bool Graph::loadRowPointers(const std::string& filePath, std::vector<int>& vec) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire row_pointers: " << filePath << std::endl;
        return false;
    }
    
    vec.reserve(num_nodes + 1);
    int val;
    while (file >> val) {
        vec.push_back(val);
    }
    file.close();
    return true;
}

//Lettura delle Column Indices (Interi, uno per riga)
bool Graph::loadColumnIndices(const std::string& filePath, std::vector<int>& vec) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire column_indices: " << filePath << std::endl;
        return false;
    }
    
    vec.reserve(num_edges);
    int val;
    while (file >> val) {
        vec.push_back(val);
    }
    file.close();
    return true;
}

//Lettura delle Node Features (Float, uno per riga, linearizzati)
bool Graph::loadFeatures(const std::string& filePath, std::vector<float>& vec) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire features: " << filePath << std::endl;
        return false;
    }
    
    vec.reserve(num_nodes * feature_dim);
    float val;
    while (file >> val) {
        vec.push_back(val);
    }
    file.close();
    return true;
}

//Lettura dei Labels (Interi, uno per riga)
bool Graph::loadLabels(const std::string& filePath, std::vector<int>& vec) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Errore: impossibile aprire labels: " << filePath << std::endl;
        return false;
    }
    
    vec.reserve(num_nodes);
    int val;
    while (file >> val) {
        vec.push_back(val);
    }
    file.close();
    return true;
}


// Funzione load principale
bool Graph::loadGraph(const std::string& folderPath) {
    //Caricamento dei metadati per dimensionare le strutture
    if (!loadMetadata(folderPath + "/metadata.txt")) return false;

    std::cout << "Metadata caricati. Nodi: " << num_nodes << ", Archi: " << num_edges << ", Dim. Feature: " << feature_dim << std::endl;

    //Caricamento delle altre strutture dati
    if (!loadRowPointers(folderPath + "/row_pointers.txt", row_pointers)) return false;
    if (!loadColumnIndices(folderPath + "/column_indices.txt", column_indices)) return false;
    if (!loadFeatures(folderPath + "/node_features.txt", node_features)) return false;
    if (!loadLabels(folderPath + "/labels.txt", labels)) return false;

    return true;
}

std::vector<int> Graph::getNeighbors(int v) const {
    // Controllo di sicurezza sull'indice del nodo
    if (v < 0 || v >= num_nodes) {
        std::cerr << "Errore: Indice nodo v fuori dai limiti (" << v << ")" << std::endl;
        return {};
    }

    //I vicini del nodo v si trovano nell'intervallo row_pointers[v] e row_pointers[v + 1] all'interno di column_indices
    int start_idx = row_pointers[v];
    int end_idx = row_pointers[v + 1];

    return std::vector<int>(column_indices.begin() + start_idx, column_indices.begin() + end_idx);
}

VertexData Graph::getVertex(int v) const {
    VertexData vertex_data;

    // Controllo di sicurezza sull'indice del nodo
    if (v < 0 || v >= num_nodes) {
        std::cerr << "Errore: Indice nodo v fuori dai limiti (" << v << ")" << std::endl;
        return vertex_data;
    }

    // Le feature del nodo v iniziano a (v * feature_dim) e finiscono a ((v + 1) * feature_dim)
    int start_feat = v * feature_dim;
    int end_feat = start_feat + feature_dim;

    vertex_data.features = std::vector<float>(node_features.begin() + start_feat, node_features.begin() + end_feat);
    vertex_data.label = labels[v];
    vertex_data.neighbors = getNeighbors(v);

    return vertex_data;
}


//Test
#define TEST 0
#if TEST
std::string folder = "../../../dataset/converted/Cora";
int main() {
    Graph g;

    //--- Test loadGraph ---
    if (g.loadGraph(folder)) {
        std::cout << "Grafo caricato con successo!" << std::endl;
        std::cout << "Dimensione Row Pointers effettiva: " << g.getRowPointers().size() << std::endl;
        std::cout << "Dimensione Column Indices effettiva: " << g.getColumnIndices().size() << std::endl;
        std::cout << "Dimensione Node Features effettiva: " << g.getNodeFeatures().size() << std::endl;
        std::cout << "Dimensione Labels effettiva: " << g.getLabels().size() << std::endl;

        // --- Test getNeighbors ---
        int target_node = 0;
        std::vector<int> neighbors = g.getNeighbors(target_node);
        std::cout << "\nVicini del nodo " << target_node << ": ";
        for (int neighbor : neighbors) {
            std::cout << neighbor << " ";
        }
        std::cout << "\n(Totale vicini: " << neighbors.size() << ")" << std::endl;

        // --- Test getVertex ---
        auto vertex_data = g.getVertex(target_node);
        std::cout << "\nLabel del nodo " << target_node << ": " << vertex_data.label << std::endl;
        std::cout << "Dimensione delle feature estratte: " << vertex_data.features.size() << std::endl;
        std::cout << "Vicini restituiti: " << vertex_data.neighbors.size() << std::endl;
        
        // Stampa delle prime 5 feature
        std::cout << "Prime 5 feature: ";
        for (size_t i = 0; i < std::min(vertex_data.features.size(), size_t(5)); ++i) {
            std::cout << vertex_data.features[i] << " ";
        }
        std::cout << "..." << std::endl;
    } else {
        std::cerr << "Errore durante il caricamento del grafo." << std::endl;
    }

    std::cout << "\nFine" << std::endl;
    std::cin.get();
    return 0;
}

#endif
