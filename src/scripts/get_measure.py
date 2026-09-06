import subprocess
import os
import sys
import itertools

# --- CONFIGURAZIONE ---
EXE_DIR = "../GCN/sequential_dense"
EXE_NAME = "./program"
OUTPUT_TXT = "benchmark_results.txt"
DATASET_DIR = "../../dataset/converted"
SEED = 42
NUM_RUNS = 3
INCLUDE_CHECKSUMS = True  # Se uguale a True considera probability_checksum e prediction_checksum

# --- COMBINAZIONI ---
datasets = ["Cora","ErdosRenyi_5k"] 
hidden_dims = [64]
layers_list = [2]

def get_dataset_info(dataset_name):
    meta_path = os.path.join(DATASET_DIR, dataset_name, "metadata.txt")
    if not os.path.exists(meta_path):
        print(f"Errore critico: file di metadata non trovato in {meta_path}.")
        sys.exit(1)
        
    info = {}
    with open(meta_path, "r") as f:
        for line in f:
            parts = line.strip().split(":")
            if len(parts) == 2:
                info[parts[0].strip()] = parts[1].strip()
                
    # Aggiunto 'num_classes' ai campi obbligatori da verificare
    for key in ["num_nodes", "num_edges", "feature_dim", "num_classes"]:
        if key not in info:
            print(f"Errore critico: metadato mancante '{key}' nel file {meta_path}.")
            sys.exit(1)
            
    desc = f"(Nodi: {info['num_nodes']}, Archi: {info['num_edges']}, Dim. Feature: {info['feature_dim']})"
    # Restituiamo sia la stringa descrittiva che il numero di classi convertito a intero
    return desc, int(info['num_classes'])

if __name__ == "__main__":
    print(f"Inizio benchmark automatico. Output in: {OUTPUT_TXT}")
    
    with open(OUTPUT_TXT, "a") as f_txt:
        for dataset, hidden, layers in itertools.product(datasets, hidden_dims, layers_list):
            info_nodi, classes = get_dataset_info(dataset)
            
            print(f"Eseguo {NUM_RUNS} run per: {dataset} | Hidden: {hidden} | Livelli: {layers}...")
            
            # Dizionari per salvare le metriche
            avg_metrics = {
                "inference_ms": 0.0,
                "graph_nodes_per_second": 0.0,
                "node_updates_per_second": 0.0,
                "messages_per_second": 0.0
            }
            
            static_metrics = {}
            implementation_mode = ""
            
            for run in range(NUM_RUNS):
                pesi_dir = f"../../../weights/{dataset}/h{hidden}_l{layers}_seed{SEED}"
                
                cmd = [
                    EXE_NAME, str(dataset), str(hidden), str(classes), str(layers), pesi_dir
                ]

                proc = subprocess.run(cmd, cwd=EXE_DIR, capture_output=True, text=True)
                
                if proc.returncode != 0:
                    print(f"\n[!] Errore durante l'esecuzione di: {' '.join(cmd)}")
                    print(f"Stdout:\n{proc.stdout}")
                    sys.exit(1)

                lines = proc.stdout.strip().split("\n")
                
                for line in lines:
                    if "=" not in line:
                        continue
                    
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    
                    if key == "RESULT implementation":
                        implementation_mode = val
                    elif key in avg_metrics:
                        avg_metrics[key] += float(val)
                    elif "memory_bytes" in key:
                        static_metrics[key] = val
                    elif "checksum" in key:
                        # Se la flag è attiva salviamo il checksum, altrimenti lo ignoriamo
                        if INCLUDE_CHECKSUMS:
                            static_metrics[key] = val
            
            # Calcolo media matematica
            for key in avg_metrics:
                avg_metrics[key] /= NUM_RUNS
            
            # Salvataggio
            f_txt.write(f"---------- Descrizione Configurazione ----------\n")
            f_txt.write(f"Dataset: {dataset} {info_nodi}\n")
            f_txt.write(f"Hidden dimension: {hidden}\n")
            f_txt.write(f"Classi: {classes}\n")
            f_txt.write(f"Livelli: {layers}\n")
            f_txt.write(f"Modalita': {implementation_mode}\n")
            f_txt.write(f"---------- Risultati Medi ({NUM_RUNS} runs) ----------\n")
            
            for key, val in avg_metrics.items():
                # Formattiamo a 2 cifre decimali, tranne per i throughput che non hanno senso decimale
                if "per_second" in key:
                    f_txt.write(f"{key}={int(round(val))}\n")
                else:
                    f_txt.write(f"{key}={val:.2f}\n")
                
            for key, val in static_metrics.items():
                f_txt.write(f"{key}={val}\n")
                
            f_txt.write("\n\n")
            f_txt.flush()

    print(f"\nBenchmark completato con successo! File aggiornato in: {OUTPUT_TXT}")