import subprocess
import os
import sys
import itertools

# --- CONFIGURAZIONE ---
EXE_DIR = "../GCN/CUDA/edge_parallelization/basic_version"
EXE_NAME = "./program"
OUTPUT_TXT = "gpu_profiling_results.txt"
DATASET_DIR = "../../dataset/converted"
SEED = 42

# --- COMBINAZIONI ---
datasets = ["Cora", "BarabasiAlbert_100k", "ogbn-arxiv"] 
hidden_dims = [64]
layers_list = [2]

# Metriche richieste da Nsight Compute
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
                
    for key in ["num_nodes", "num_edges", "feature_dim", "num_classes"]:
        if key not in info:
            print(f"Errore critico: metadato mancante '{key}' nel file {meta_path}.")
            sys.exit(1)
            
    desc = f"(Nodi: {info['num_nodes']}, Archi: {info['num_edges']}, Dim. Feature: {info['feature_dim']})"
    return desc, int(info['num_classes'])

if __name__ == "__main__":
    print(f"Inizio profilazione GPU con Nsight Compute. Output in: {OUTPUT_TXT}")
    
    with open(OUTPUT_TXT, "a") as f_txt:
        for dataset, hidden, layers in itertools.product(datasets, hidden_dims, layers_list):
            
            info_nodi, classes = get_dataset_info(dataset)
            
            f_txt.write(f"---------- Parametri ----------\n")
            f_txt.write(f"Dataset: {dataset} {info_nodi}\n")
            f_txt.write(f"Hidden dimension: {hidden}\n")
            f_txt.write(f"Classi: {classes}\n")
            f_txt.write(f"Livelli: {layers}\n")
            f_txt.write(f"Target: CUDA GPU\n")
            f_txt.write(f"---------- Esecuzioni ----------\n")

            print(f"Eseguo: {dataset} | Hidden: {hidden} | Livelli: {layers}...")
            
            # Attenzione al percorso dei pesi: in CUDA solitamente si trova a 5 livelli di profondità
            pesi_dir = f"../../../../../weights/{dataset}/h{hidden}_l{layers}_seed{SEED}"
            
           cmd = [
                "ncu",
                "--launch-count", "1",
                "--metrics", metrics_str,
                EXE_NAME, str(dataset), str(hidden), str(classes), str(layers), pesi_dir
            ]
            
            proc = subprocess.run(cmd, cwd=EXE_DIR, capture_output=True, text=True)
            output_lines = proc.stdout.strip().split("\n")
            
            if proc.returncode != 0 and len(output_lines) <= 1:
                print(f"\n[!] Errore durante l'esecuzione di: {' '.join(cmd)}")
                print(f"Codice uscita: {proc.returncode}")
                print(f"Stderr:\n{proc.stderr}")
                sys.exit(1)
            
            # Estrazione dinamica dell'implementazione
            implementation_mode = "unknown"
            for out_line in output_lines:
                if out_line.startswith("RESULT implementation="):
                    implementation_mode = out_line.split("=", 1)[1].strip()
                    break

            # Dizionari per accumulare i valori dei kernel (ncu ne stampa uno per ogni lancio)
            accumulators = {m: [] for m in METRICS}
            
            # Parsing delle metriche ncu
            for line in output_lines:
                line_lower = line.lower()
                for m in METRICS:
                    if m.lower() in line_lower:
                        parts = line.strip().split()
                        for part in reversed(parts):
                            try:
                                val = float(part.replace(",", ""))
                                accumulators[m].append(val)
                                break
                            except ValueError:
                                continue

            def avg_metric(m_name):
                vals = accumulators[m_name]
                return round(sum(vals) / len(vals), 2) if vals else 0.0

            occupancy = avg_metric("sm__warps_active.avg.pct_of_peak_sustained_active")
            dram_thr  = avg_metric("dram__throughput.avg.pct_of_peak_sustained_elapsed")
            l2_hit    = avg_metric("lts__t_sector_hit_rate.pct")
            sm_thr    = avg_metric("sm__throughput.avg.pct_of_peak_sustained_elapsed")
            ipc_sm    = avg_metric("sm__inst_executed.avg.per_cycle_active")
            
            f_txt.write("========================================\n")
            f_txt.write(f"RESULT implementation:{implementation_mode}-profiling\n")
            f_txt.write(f"Achieved Occupancy:{occupancy}%\n")
            f_txt.write(f"DRAM Throughput:{dram_thr}%\n")
            f_txt.write(f"L2 Cache Hit Rate:{l2_hit}%\n")
            f_txt.write(f"SM Throughput:{sm_thr}%\n")
            f_txt.write(f"IPC (per SM):{ipc_sm}\n\n\n")
            f_txt.flush()

    print(f"\nProfilazione GPU completata! File aggiornato in: {OUTPUT_TXT}")