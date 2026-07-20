import os
import numpy as np

dir_dataset_out = "../../dataset/converted"

def export_dataset(dir_output, row_pointers, column_indices, node_features, labels, num_nodes, num_edges, feature_dim):
    # Salvataggio in file testuali
    os.makedirs(f"{dir_dataset_out}/{dir_output}", exist_ok=True)
    
    np.savetxt(f"{dir_dataset_out}/{dir_output}/row_pointers.txt", row_pointers, fmt="%d")
    np.savetxt(f"{dir_dataset_out}/{dir_output}/column_indices.txt", column_indices, fmt="%d")
    np.savetxt(f"{dir_dataset_out}/{dir_output}/node_features.txt", node_features.flatten(), fmt="%.6f")
    np.savetxt(f"{dir_dataset_out}/{dir_output}/labels.txt", labels, fmt="%d")
    
    # Metadati utili
    with open(f"{dir_dataset_out}/{dir_output}/metadata.txt", "w") as f:
        f.write(f"num_nodes:{num_nodes}\n")
        f.write(f"num_edges:{num_edges}\n")
        f.write(f"feature_dim:{feature_dim}\n")

    print("\n Dataset esportato con successo")
    print("Struttura dei file generati:")
    print(" - row_pointers.txt   -> Puntatori di riga (CSR)")
    print(" - column_indices.txt -> Indici di colonna ordinati (CSR)")
    print(" - node_features.txt  -> Matrice densa delle feature linearizzata")
    print(" - labels.txt         -> Target per la classificazione")