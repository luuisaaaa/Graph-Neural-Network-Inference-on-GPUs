import subprocess
import os
import sys
import itertools

# --- CONFIGURAZIONE ---
EXE_DIR = "../GCN/sequential_dense"
EXE_NAME = "./program"
OUTPUT_TXT = "profiling_results.txt"
DATASET_DIR = "../../dataset/converted"
SEED = 42

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
                
    for key in ["num_nodes", "num_edges", "feature_dim", "num_classes"]:
        if key not in info:
            print(f"Errore critico: metadato mancante '{key}' nel file {meta_path}.")
            sys.exit(1)
            
    desc = f"(Nodi: {info['num_nodes']}, Archi: {info['num_edges']}, Dim. Feature: {info['feature_dim']})"
    return desc, int(info['num_classes'])

if __name__ == "__main__":
    print(f"Inizio profilazione CPU. Output in: {OUTPUT_TXT}")
    
    with open(OUTPUT_TXT, "a") as f_txt:
        for dataset, hidden, layers in itertools.product(datasets, hidden_dims, layers_list):
            
            info_nodi, classes = get_dataset_info(dataset)
            
            f_txt.write(f"---------- Parametri ----------\n")
            f_txt.write(f"Dataset: {dataset} {info_nodi}\n")
            f_txt.write(f"Hidden dimension: {hidden}\n")
            f_txt.write(f"Classi: {classes}\n")
            f_txt.write(f"Livelli: {layers}\n")
            f_txt.write(f"---------- Esecuzioni ----------\n")

            print(f"Eseguo: {dataset} | Hidden: {hidden} | Livelli: {layers}...")
            
            pesi_dir = f"../../../weights/{dataset}/h{hidden}_l{layers}_seed{SEED}"
            
            cmd = [
                "perf", "stat", "-x,", "-i",
                "-e", "cycles,instructions,L1-dcache-loads,L1-dcache-load-misses",
                EXE_NAME, str(dataset), str(hidden), str(classes), str(layers), pesi_dir
            ]

            proc = subprocess.run(cmd, cwd=EXE_DIR, capture_output=True, text=True)
            
            if proc.returncode != 0:
                print(f"\n[!] Errore durante l'esecuzione di: {' '.join(cmd)}")
                print(f"Codice uscita: {proc.returncode}")
                print(f"Stdout:\n{proc.stdout}")
                print(f"Stderr:\n{proc.stderr}")
                sys.exit(1)

            # Estrazione dinamica del nome dell'implementazione
            implementation_mode = "unknown"
            for out_line in proc.stdout.strip().split("\n"):
                if out_line.startswith("RESULT implementation="):
                    implementation_mode = out_line.split("=", 1)[1].strip()
                    break

            # Estrazione delle metriche di perf
            lines = proc.stderr.strip().split("\n")
            cycles, instr, l1_loads, l1_misses = 0, 0, 0, 0
            
            for line in lines:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 2:
                    continue
                
                raw_line = line.lower()
                val_str = parts[0]
                num = int(val_str) if val_str.isdigit() else 0
                
                if "cycles" in raw_line and cycles == 0:
                    cycles = num
                elif "instructions" in raw_line and instr == 0:
                    instr = num
                elif "l1-dcache-loads" in raw_line and l1_loads == 0:
                    l1_loads = num
                elif "l1-dcache-load-misses" in raw_line and l1_misses == 0:
                    l1_misses = num
            
            ipc = round(instr / cycles, 2) if cycles else 0.0
            l1_rate = round((l1_misses / l1_loads) * 100, 2) if l1_loads else 0.0
            
            f_txt.write("========================================\n")
            f_txt.write(f"RESULT implementation:{implementation_mode}-profiling\n")
            f_txt.write(f"Cycles:{cycles}\n")
            f_txt.write(f"Instructions:{instr}\n")
            f_txt.write(f"IPC:{ipc}\n")
            f_txt.write(f"L1 Cache Loads:{l1_loads}\n")
            f_txt.write(f"L1 Cache Misses:{l1_misses}\n")
            f_txt.write(f"L1 Miss Rate:{l1_rate}%\n\n\n")
            f_txt.flush()

    print(f"\nProfilazione CPU completata! File aggiornato in: {OUTPUT_TXT}")