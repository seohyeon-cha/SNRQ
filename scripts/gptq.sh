MODEL_PATH="meta-llama/Llama-2-7b-hf"

ALPHA=0.25
BETA=0.0003

WBITS_VALUES=(3)
SEEDS=(0)

DATE=$(date +"%Y%m%d")
mkdir -p logs/l2-7b

for wbits in "${WBITS_VALUES[@]}"; do
  echo "===== Testing bitwidth wbits = $wbits ====="
  
  for seed in "${SEEDS[@]}"; do
    echo "--- Running seed = $seed ---"
    
    LOG_FILE="ogs/l2-7b/${DATE}_l2-7b-${wbits}bit-128g_gptq_seed${seed}.log"
    
    CMD="CUDA_VISIBLE_DEVICES=0 python -u llama_step.py \
      $MODEL_PATH c4 \
      --method gptq \
      --wbits $wbits \
      --true-sequential \
      --sym \
      --act-order \
      --groupsize 128 \
      --seed $seed \
      --eval \
      --lm-eval \
      --wandb \
      --wandb-project L2-7B \
      --wandb-name gptq_${wbits}bit_seed${seed}"
    
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

