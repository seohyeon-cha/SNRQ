#!/bin/bash
#SBATCH -J fp16        # Job name
#SBATCH -p gh             # Partition (queue) name
#SBATCH -N 1              # Total number of nodes
#SBATCH -n 1              # Total number of MPI tasks
#SBATCH -t 0:30:00     
#SBATCH --output=slurm_out/fp16_%j.out
#SBATCH --error=slurm_out/fp16_%j.err
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

MODEL_PATH="meta-llama/Meta-Llama-3-8B"

SEEDS=(0 1 2)

DATE=$(date +"%Y%m%d")
cd /work/10322/scha0901/vista/FOEM/LLM/weight-only
mkdir -p logs/l3-8b
mkdir -p slurm_out

echo "===== Running FP16 (Full Precision) Evaluation ====="

for seed in "${SEEDS[@]}"; do
  echo "--- Running seed = $seed ---"
  
  LOG_FILE="/work/10322/scha0901/vista/FOEM/LLM/weight-only/logs/l3-8b/${DATE}_l3-8b-fp16_seed${seed}.log"
  
  # Use SLURM GPU if available, otherwise use GPU 0
  GPU_ID=${SLURM_LOCALID:-0}
  CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $VENV/bin/python -u /work/10322/scha0901/vista/FOEM/LLM/weight-only/llama_step.py \
    $MODEL_PATH c4 \
    --wbits 16 \
    --true-sequential \
    --seed $seed \
      --eval \
      --lm-eval \
      --wandb \
      --wandb-project L3-8B-symm-new \
    --wandb-name fp16_seed${seed}"
  
  echo "Executing command: $CMD"
  echo "Log will be saved to: $LOG_FILE"
  
  eval "$CMD" | tee "$LOG_FILE"
  
  echo "--- Seed $seed evaluation completed ---"
  echo ""
  
  nvidia-smi --gpu-reset 2>/dev/null || true
  sleep 10
done

echo "All FP16 evaluations completed!"

