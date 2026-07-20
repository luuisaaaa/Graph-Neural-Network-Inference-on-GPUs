import sys
import numpy as np
from scipy.sparse import csr_matrix
from torch_geometric.datasets import Planetoid
from converter import export_dataset

dir_dataset_in = "../../dataset/raw"

def load_dataset(dataset_name):
    dir_output = dataset_name
    print(f"\nCaricamento del dataset {dataset_name} in corso...")
    
    # Scarica il dataset nella cartella definita se non presente
    dataset = Planetoid(root=f"{dir_dataset_in}", name=dataset_name)
    data = dataset[0]

    # Informazioni generali sul grafo
    num_nodes = data.num_nodes
    num_edges = data.num_edges
    
    # data.edge_index contiene gli archi in formato COO (matrice 2xM)
    # Rispettivamente: array dei nodi sorgente e array dei nodi destinazione
    edges = data.edge_index.numpy()
    src_nodes = edges[0]
    dst_nodes = edges[1]
    
    print(f"-> Nodi rilevati: {num_nodes}")
    print(f"-> Archi rilevati: {num_edges}")

    # Conversione in CSR tramite SciPy per ottenere l'ordinamento topologico
    dummy_data = np.ones(num_edges, dtype=np.int8)
    matrix_coo = csr_matrix((dummy_data, (src_nodes, dst_nodes)), shape=(num_nodes, num_nodes))
    matrix_csr = matrix_coo.tocsr()

    # Estrazione vettori CSR pronti per il C++
    row_pointers = matrix_csr.indptr.astype(np.int32)      # Array Row Index (dimensione V + 1)
    column_indices = matrix_csr.indices.astype(np.int32)   # Array Column Index (dimensione E, ordinato)

    # Estrazione delle feature dei nodi e delle etichette
    # data.x è una matrice densa (|V| x F).
    node_features = data.x.numpy().astype(np.float32)
    feature_dim = node_features.shape[1]
    print(f"-> Dimensione feature vector (F): {feature_dim}")
    
    # data.y contiene le classi reali dei nodi
    labels = data.y.numpy().astype(np.int32)

    # Esportazione senza pesi
    export_dataset(dir_output, row_pointers, column_indices, node_features, labels, num_nodes, num_edges, feature_dim)
    print(f"Esportazione di {dataset_name} completata con successo!\n")


def main():
    valid_datasets = {
        'cora': 'Cora',
        'citeseer': 'CiteSeer',
        'pubmed': 'PubMed'
    }
    
    if len(sys.argv) < 2:
        print("Errore: Nessun dataset specificato.")
        print("Uso corretto: python script.py [Cora|CiteSeer|PubMed]")
        sys.exit(1)
        
    option = sys.argv[1].lower()
    
    if option in valid_datasets:
        name_dataset = valid_datasets[option]
        load_dataset(name_dataset)
    else:
        print(f"Errore: Dataset '{sys.argv[1]}' non valido.")
        print("I dataset supportati sono: Cora, CiteSeer, PubMed.")
        sys.exit(1)

if __name__ == "__main__":
    main()