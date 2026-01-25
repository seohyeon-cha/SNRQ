MODEL_PATH="meta-llama/Llama-2-7b-hf"
BETA=0.0003
WBITS_VALUES=(3)
SEEDS=(0)

DATE=$(date +"%Y%m%d")
mkdir -p logs/l2-7b

for wbits in "${WBITS_VALUES[@]}"; do
  echo "===== Testing bitwidth wbits = $wbits ====="
  
  for seed in "${SEEDS[@]}"; do
    echo "--- Running seed = $seed ---"
    
    LOG_FILE="logs/l2-7b/${DATE}_l2-7b-${wbits}bit-128g_ldlq_seed${seed}.log"
    
    CMD="CUDA_VISIBLE_DEVICES=0 python -u llama_step.py \
      $MODEL_PATH c4 \
      --method ldlq \
      --sym \
      --wbits $wbits \
      --true-sequential \
      --groupsize 128 \
      --seed $seed \
      --nsamples 128 \
      --eval \
      --lm-eval \
      --wandb \
      --wandb-project L2-7B \
      --wandb-name ldlq_${wbits}bit_seed${seed}"
    
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

