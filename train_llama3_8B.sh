#!/bin/bash

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

MODEL_PATH="meta-llama/Meta-Llama-3-8B-Instruct"


ALPHA=0.25
BETA=0.0003

WBITS_VALUES=(16)


DATE=$(date +"%Y%m%d")


mkdir -p logs6

declare -A METHODS=(
    # ["gptaq"]="--method gptaq --alpha ${ALPHA}"
    # ["greedyaq"]="--method greedyaq --alpha ${ALPHA}"
    # ["foem"]="--method foem --alpha ${ALPHA}"
    ["gptq"]="--method gptq"
)

for wbits in "${WBITS_VALUES[@]}"; do
  echo "===== Testing bitwidth wbits = $wbits ====="
  
  for method_name in "${!METHODS[@]}"; do
    method_args="${METHODS[$method_name]}"
    
    echo "--- Running method: $method_name ---"
    
    LOG_FILE="logs6/${DATE}_llama3-8b-${wbits}bit-128g_${method_name}.log"
    
    CMD="CUDA_VISIBLE_DEVICES=0 $VENV/bin/python -u llama_step.py \
      $MODEL_PATH c4 \
      --wbits $wbits \
      --true-sequential \
      --act-order \
      --groupsize 128 \
      --eval \
      --lm-eval \
      $method_args"
    
    echo "Executing command: $CMD"
    echo "Log will be saved to: $LOG_FILE"
    
    eval "$CMD" | tee "$LOG_FILE"
    
    echo "--- Method $method_name evaluation completed ---"
    echo ""
    
    nvidia-smi --gpu-reset 2>/dev/null || true
    sleep 10
  done
  
  echo "===== All methods for bitwidth wbits = $wbits completed ====="
  echo ""
done

echo "All tests completed!"