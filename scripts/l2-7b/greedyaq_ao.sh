#!/bin/bash
#SBATCH -J greedyaq        # Job name
#SBATCH -p gh             # Partition (queue) name
#SBATCH -N 1              # Total number of nodes
#SBATCH -n 1              # Total number of MPI tasks
#SBATCH -t 3:00:00     
#SBATCH --output=slurm_out/greedyaq_%j.out
#SBATCH --error=slurm_out/greedyaq_%j.err
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
ALPHA=0.2
ALPHA_METHOD="alternate"
BETA=0.0003
WBITS_VALUES=(3)
SEEDS=(0 1 2)

DATE=$(date +"%Y%m%d") 


for wbits in "${WBITS_VALUES[@]}"; do
  echo "===== Testing bitwidth wbits = $wbits ====="
  
  for seed in "${SEEDS[@]}"; do
    echo "--- Running seed = $seed ---"
    
    LOG_FILE="logs/l2-7b/${DATE}_l2-7b-${wbits}bit-128g_greedyaq_seed${seed}.log"
    
    CMD="CUDA_VISIBLE_DEVICES=0 $VENV/bin/python -u llama_step.py \
      $MODEL_PATH c4 \
      --method greedyaq \
      --sym \
      --wbits $wbits \
      --true-sequential \
      --groupsize 128 \
      --seed $seed \
      --alpha-method ${ALPHA_METHOD} \
      --alpha ${ALPHA} \
      --nsamples 256 \
      --eval \
      --lm-eval \
      --wandb-project L2-7B-KD-test \
      --wandb-name greedyaq_${wbits}bit_seed${seed}"
    
    echo "Executing command: $CMD"
    echo "Log will be saved to: $LOG_FILE"
    
    eval "$CMD" | tee "$LOG_FILE"
    
    echo "--- Seed $seed evaluation completed ---"
    echo ""
    
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 10
  done
  
  echo "===== All seeds for bitwidth wbits = $wbits completed ====="
  echo ""
done

echo "All tests completed!"

