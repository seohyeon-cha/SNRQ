#!/bin/bash
#SBATCH -J greedyaq_multi_alpha        # Job name
#SBATCH -p gh             # Partition (queue) name
#SBATCH -N 1              # Total number of nodes
#SBATCH -n 1              # Total number of MPI tasks
#SBATCH -t 2:30:00     
#SBATCH --output=slurm_out/greedyaq_multi_alpha_%j.out
#SBATCH --error=slurm_out/greedyaq_multi_alpha_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=seohyeon.cha@utexas.edu

VENV="$WORK/venvs/gptaq"
source "$VENV/bin/activate"

# 2) If pip is still missing inside the venv, bootstrap it
"$VENV/bin/python" -m ensurepip --upgrade
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
alias pip='"$VENV/bin/python" -m pip'

# 3) Point to your local CUDA 12.4 install (headers confirmed in include/)
export CUDA_HOME="$WORK/cuda-12.4"
test -f "$CUDA_HOME/include/cusparse.h" 

# aarch64 libs; add lib64 if present
export LD_LIBRARY_PATH="$CUDA_HOME/targets/aarch64-linux/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"
[ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# prefer system GCC 11 over module gcc 15
export PATH=/usr/bin:$CUDA_HOME/bin:$PATH
hash -r
which gcc; gcc --version
export CC=/usr/bin/gcc; export CXX=/usr/bin/g++; export CUDAHOSTCXX=/usr/bin/g++

# sanity check: must show cu124 and CUDA available True
"$VENV/bin/python" - <<'PY'
import torch, sys
print("Torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("exe:", sys.executable)
PY

cd /work/10322/scha0901/vista/FOEM/LLM/weight-only

MODEL_PATH="meta-llama/Llama-2-7b-hf"
ALPHA_METHOD="fixed"
BETA=0.0003
WBITS_VALUES=(3)
SEEDS=(0)  # Use only seed 0 for faster execution, or use (0 1 2) for full runs
ALPHA_VALUES=(0.0 0.25 0.5 0.75 1.0)  # Multiple alpha values to test

DATE=$(date +"%Y%m%d") 
BASE_PLOT_PATH="plots/delta_x/l2-7b/greedyaq_multiple_alphas"
RESULTS_LOG="${BASE_PLOT_PATH}/results_table_${DATE}.log"

# Initialize results log file
mkdir -p "${BASE_PLOT_PATH}"
echo "Results Table - Multiple Alpha Comparison" > "$RESULTS_LOG"
echo "Date: $(date)" >> "$RESULTS_LOG"
echo "Method: greedyaq" >> "$RESULTS_LOG"
echo "WBits: ${WBITS_VALUES[@]}" >> "$RESULTS_LOG"
echo "Seeds: ${SEEDS[@]}" >> "$RESULTS_LOG"
echo "Alpha Values: ${ALPHA_VALUES[@]}" >> "$RESULTS_LOG"
echo "" >> "$RESULTS_LOG"
echo "=" >> "$RESULTS_LOG"
echo "" >> "$RESULTS_LOG"
echo "RESULTS TABLE" >> "$RESULTS_LOG"
echo "=" >> "$RESULTS_LOG"
echo "" >> "$RESULTS_LOG"
echo "Alpha      Seed     Wikitext2 PPL   C4 PPL          Avg Acc" >> "$RESULTS_LOG"
echo "----------------------------------------------------------------------------------------------------" >> "$RESULTS_LOG"

# Run quantization for each alpha
for wbits in "${WBITS_VALUES[@]}"; do
  echo "===== Testing bitwidth wbits = $wbits ====="
  
  for alpha in "${ALPHA_VALUES[@]}"; do
    echo "===== Running alpha = $alpha ====="
    
    for seed in "${SEEDS[@]}"; do
      echo "--- Running seed = $seed ---"
      
      LOG_FILE="logs/l2-7b/${DATE}_l2-7b-${wbits}bit-128g_greedyaq_alpha${alpha}_seed${seed}.log"
      
      CMD="CUDA_VISIBLE_DEVICES=0 $VENV/bin/python -u llama_step.py \
        $MODEL_PATH c4 \
        --method greedyaq \
        --sym \
        --wbits $wbits \
        --true-sequential \
        --groupsize 128 \
        --seed $seed \
        --alpha-method ${ALPHA_METHOD} \
        --alpha ${alpha} \
        --nsamples 128 \
        --eval \
        --lm-eval \
        --plot-delta-x \
        --plot-delta-x-path ${BASE_PLOT_PATH}/alpha${alpha} \
        --wandb-project L2-7B-symm-new \
        --wandb-name greedyaq_${wbits}bit_alpha${alpha}_seed${seed}"
      
      echo "Executing command: $CMD"
      echo "Log will be saved to: $LOG_FILE"
      
      eval "$CMD" | tee "$LOG_FILE"
      
      echo "--- Seed $seed evaluation completed for alpha $alpha ---"
      echo ""
      
      # Parse and write results immediately after each run
      "$VENV/bin/python" <<PYTHON_PARSE
import sys
sys.path.insert(0, '/work/10322/scha0901/vista/FOEM/LLM/weight-only')
import os
import re
from datetime import datetime

log_file = '${LOG_FILE}'
results_log = '${RESULTS_LOG}'
alpha = ${alpha}
seed = ${seed}
wbits = ${wbits}

if os.path.exists(log_file):
    with open(log_file, 'r') as lf:
        content = lf.read()
        
        # Extract perplexity
        wikitext2_ppl = 'N/A'
        c4_ppl = 'N/A'
        
        # Pattern: "wikitext2" followed by "Evaluating" then number
        match = re.search(r'wikitext2\\s*\\n.*?Evaluating.*?\\n\\s*([0-9]+\\.[0-9]+)', content, re.MULTILINE | re.DOTALL)
        if match:
            wikitext2_ppl = match.group(1)
        else:
            # Fallback: "wikitext2" followed by number within next 3 lines
            match = re.search(r'wikitext2[^\\n]*\\n[^\\n]*\\n[^\\n]*\\n\\s*([0-9]+\\.[0-9]+)', content, re.MULTILINE)
            if match:
                wikitext2_ppl = match.group(1)
        
        # Pattern: "c4" followed by "Evaluating" then number
        match = re.search(r'\\bc4\\s*\\n.*?Evaluating.*?\\n\\s*([0-9]+\\.[0-9]+)', content, re.MULTILINE | re.DOTALL)
        if match:
            c4_ppl = match.group(1)
        else:
            # Fallback: "c4" followed by number within next 3 lines
            match = re.search(r'\\bc4[^\\n]*\\n[^\\n]*\\n[^\\n]*\\n\\s*([0-9]+\\.[0-9]+)', content, re.MULTILINE)
            if match:
                c4_ppl = match.group(1)
        
        # Extract lm_eval results - only avg_acc
        avg_acc = 'N/A'
        # Try to find avg_acc or acc_avg in the results
        match = re.search(r'[\'"]avg_acc[\'"]\\s*:\\s*([0-9]+\\.[0-9]+)', content)
        if match:
            avg_acc = match.group(1)
        else:
            # Try acc_avg
            match = re.search(r'[\'"]acc_avg[\'"]\\s*:\\s*([0-9]+\\.[0-9]+)', content)
            if match:
                avg_acc = match.group(1)
        
        # Write to results log immediately
        with open(results_log, 'a') as f:
            # Write the result row
            f.write(f"{alpha:<10} {seed:<8} {wikitext2_ppl:<15} {c4_ppl:<15} {avg_acc:<15}\\n")
            f.flush()  # Ensure it's written immediately
        
        print(f'Results written to log: alpha={alpha}, seed={seed}, wikitext2_ppl={wikitext2_ppl}, c4_ppl={c4_ppl}, avg_acc={avg_acc}')
else:
    print(f'Warning: Log file not found: {log_file}')
PYTHON_PARSE
      
      nvidia-smi --gpu-reset 2>/dev/null || true
      sleep 10
    done
    
    echo "===== Alpha $alpha completed ====="
    echo ""
  done
  
  # After all alphas are run, create combined plots
  echo "===== Creating combined plots for all alphas ====="
  "$VENV/bin/python" <<PYTHON_SCRIPT

import sys
sys.path.insert(0, '/work/10322/scha0901/vista/FOEM/LLM/weight-only')
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

base_path = '${BASE_PLOT_PATH}'
alpha_values_str = '${ALPHA_VALUES[@]}'
alpha_values = [float(x) for x in alpha_values_str.split() if x]
method_name = 'greedyaq'
results_log = '${RESULTS_LOG}'
date_str = '${DATE}'

# Load norm files for each alpha
alpha_data_mae = {}
alpha_data_fro = {}

for alpha in alpha_values:
    norm_path = f'{base_path}/alpha{alpha}/layer_norms_alpha{alpha}.pkl'
    if os.path.exists(norm_path):
        with open(norm_path, 'rb') as f:
            data = pickle.load(f)
            if 'layer_mae' in data:
                alpha_data_mae[alpha] = data['layer_mae']
            if 'layer_fro' in data:
                alpha_data_fro[alpha] = data['layer_fro']
        print(f'Loaded norms for alpha={alpha}')
    else:
        print(f'Warning: Norm file not found for alpha={alpha}: {norm_path}')

# Create combined plots
if len(alpha_data_mae) > 0 or len(alpha_data_fro) > 0:
    # Determine number of layers from first available data
    n_layers = None
    if len(alpha_data_mae) > 0:
        n_layers = len(list(alpha_data_mae.values())[0])
    elif len(alpha_data_fro) > 0:
        n_layers = len(list(alpha_data_fro.values())[0])
    
    if n_layers is not None:
        layer_indices = np.arange(n_layers)
        
        # Plot MAE norm
        if len(alpha_data_mae) > 0:
            plt.figure(figsize=(12, 6))
            for alpha in sorted(alpha_data_mae.keys()):
                plt.plot(layer_indices, alpha_data_mae[alpha], marker='o', linewidth=2, 
                        markersize=4, label=f'alpha={alpha}')
            plt.xlabel('Layer Index', fontsize=12)
            plt.ylabel('MAE |X_f - X_q|', fontsize=12)
            plt.title(f'{method_name.upper()} - MAE vs Layer Index (Multiple Alphas)', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.xlim(-0.5, n_layers - 0.5)
            plt.tight_layout()
            output_path_mae = f'{base_path}/mae_norm_multiple_alphas.png'
            plt.savefig(output_path_mae, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Combined MAE plot saved to: {output_path_mae}')
        
        # Plot Frobenius norm
        if len(alpha_data_fro) > 0:
            plt.figure(figsize=(12, 6))
            for alpha in sorted(alpha_data_fro.keys()):
                plt.plot(layer_indices, alpha_data_fro[alpha], marker='s', linewidth=2, 
                        markersize=4, label=f'alpha={alpha}')
            plt.xlabel('Layer Index', fontsize=12)
            plt.ylabel('Frobenius Norm ||X_f - X_q||_F', fontsize=12)
            plt.title(f'{method_name.upper()} - Frobenius Norm vs Layer Index (Multiple Alphas)', 
                     fontsize=14, fontweight='bold')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.4)
            plt.xlim(-0.5, n_layers - 0.5)
            plt.tight_layout()
            output_path_fro = f'{base_path}/frobenius_norm_multiple_alphas.png'
            plt.savefig(output_path_fro, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'Combined Frobenius plot saved to: {output_path_fro}')
else:
    print('Error: No data loaded for any alpha values')

# Add footer to results log
with open(results_log, 'a') as f:
    f.write('\\n')
    f.write('=' * 80 + '\\n')
    f.write(f'All results completed at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\\n')

print(f'Results table saved to: {results_log}')
PYTHON_SCRIPT
  
  echo "===== All alphas for bitwidth wbits = $wbits completed ====="
  echo ""
done

echo "All tests completed!"

