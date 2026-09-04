import subprocess
import os
import sys
import itertools

# --- CONFIGURAZIONE ---
EXE_DIR = "../GCN/CUDA/edge_parallelization/basic_version"
EXE_NAME = "./program"
OUTPUT_TXT = "profiling.txt"
DATASET_DIR = "../../dataset/converted"

# --- COMBINAZIONI ---
datasets = ["Cora", "Erdos_100k_directed"] 
hidden_dims = [64, 128]
layers_list = [2, 4]

# Numero classi finali
dataset_classes = {
    "Cora": 7,
    "ogbn-arxiv": 40,
    "ogbn-products": 47,
    "Erdos_100k_directed": 10
}

# Metriche richieste
METRICS = [
    "sm__warps_active.avg.pct_of_peak_sustained_active",  # Achieved Occupancy
    "dram__throughput.avg.pct_of_peak_sustained_elapsed", # DRAM Throughput
    "lts__t_sector_hit_rate.pct",                         # L2 Cache Hit Rate
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",   # SM / Compute Throughput
    "sm__inst_executed.avg.per_cycle_active"              # IPC (per SM)
]
metrics_str = ",".join(METRICS)


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
                
    for key in ["num_nodes", "num_edges", "feature_dim"]:
        if key not in info:
            print(f"Errore critico: metadato mancante '{key}' nel file {meta_path}.")
            sys.exit(1)
            
    return f"(Nodi: {info['num_nodes']}, Archi: {info['num_edges']}, Dim. Feature: {info['feature_dim']})"

if __name__ == "__main__":
    print(f"Inizio profilazione GPU con Nsight Compute. Output in: {OUTPUT_TXT}")
    
    with open(OUTPUT_TXT, "a") as f_txt:
        for dataset, hidden, layers in itertools.product(datasets, hidden_dims, layers_list):
            classes = dataset_classes.get(dataset, 2)
            info_nodi = get_dataset_info(dataset)
            
            f_txt.write(f"---------- Parametri ----------\n")
            f_txt.write(f"Dataset: {dataset} {info_nodi}\n")
            f_txt.write(f"Hidden dimension: {hidden}\n")
            f_txt.write(f"Classi: {classes}\n")
            f_txt.write(f"Livelli: {layers}\n")
            f_txt.write(f"Target: CUDA GPU\n")
            f_txt.write(f"---------- Esecuzioni ----------\n")

            print(f"Eseguo: {dataset} | Hidden: {hidden} | Livelli: {layers}...")
            
            # Comando ncu: output testuale standard
            cmd = [
                "ncu", "--metrics", metrics_str,
                EXE_NAME, str(dataset), str(hidden), str(classes), str(layers)
            ]
            
            proc = subprocess.run(cmd, cwd=EXE_DIR, capture_output=True, text=True)
            output_lines = proc.stdout.strip().split("\n")
            
            if proc.returncode != 0 and len(output_lines) <= 1:
                print(f"\n[!] Errore durante l'esecuzione di: {' '.join(cmd)}")
                print(f"Codice uscita: {proc.returncode}")
                print(f"Stderr del programma:\n{proc.stderr}")
                sys.exit(1)
            
            # Dizionari per accumulare i valori di tutti i kernel lanciati nell'esecuzione
            accumulators = {m: [] for m in METRICS}
            
            # Parsing testuale dell'output di ncu
            for line in output_lines:
                line_lower = line.lower()
                for m in METRICS:
                    if m.lower() in line_lower:
                        # Estrae l'ultimo elemento convertibile in float dalla riga
                        parts = line.strip().split()
                        for part in reversed(parts):
                            try:
                                val = float(part.replace(",", ""))
                                accumulators[m].append(val)
                                break
                            except ValueError:
                                continue

            # Calcolo delle medie per l'esecuzione
            def avg_metric(m_name):
                vals = accumulators[m_name]
                return round(sum(vals) / len(vals), 2) if vals else 0.0

            occupancy = avg_metric("sm__warps_active.avg.pct_of_peak_sustained_active")
            dram_thr  = avg_metric("dram__throughput.avg.pct_of_peak_sustained_elapsed")
            l2_hit    = avg_metric("lts__t_sector_hit_rate.pct")
            sm_thr    = avg_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
            ipc_sm    = avg_metric("sm__inst_executed.avg.per_cycle_active")
            
            # Scrittura dei risultati
            f_txt.write("========================================\n")
            f_txt.write("RESULT implementation:cuda-gpu-profiling\n")
            f_txt.write(f"Achieved Occupancy:{occupancy}%\n")
            f_txt.write(f"DRAM Throughput:{dram_thr}%\n")
            f_txt.write(f"L2 Cache Hit Rate:{l2_hit}%\n")
            f_txt.write(f"SM Throughput:{sm_thr}%\n")
            f_txt.write(f"IPC (per SM):{ipc_sm}\n\n\n")
            f_txt.flush()

    print(f"\nProfilazione GPU completata! File aggiornato in: {OUTPUT_TXT}")