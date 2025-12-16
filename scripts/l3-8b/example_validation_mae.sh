#!/bin/bash
#SBATCH -J validation_mae_mixup        # Job name
#SBATCH -p gh                         # Partition (queue) name
#SBATCH -N 1                          # Total number of nodes
#SBATCH -n 1                          # Total number of MPI tasks
#SBATCH -t 3:00:00
#SBATCH --output=slurm_out/validation_mae_mixup_%j.out
#SBATCH --error=slurm_out/validation_mae_mixup_%j.err

set -euo pipefail

mkdir -p slurm_out

VENV="$WORK/venvs/gptaq"
source "$VENV/bin/activate"

# Setup CUDA environment (adjust paths as needed)
export CUDA_HOME="$WORK/cuda-12.4"
export LD_LIBRARY_PATH="$CUDA_HOME/targets/aarch64-linux/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"
[ -d "$CUDA_HOME/lib64" ] && export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export PATH=/usr/bin:$CUDA_HOME/bin:$PATH

cd /work/10322/scha0901/vista/FOEM/LLM/weight-only

MODEL_PATH="meta-llama/Meta-Llama-3-8B"

# Fixed alphas (only alpha=0.5 for comparison with sampled)
FIXED_ALPHAS=(0.5)

# Sampled-alpha mixup parameters (Beta(lambda, lambda))
MIXUP_PARAMS=()

BASE_PLOT_PATH="plots/delta_x/l3-8b/validation_mae"
mkdir -p "${BASE_PLOT_PATH}"

# Set to "true" to skip quantization and only generate plots
ONLY_PLOT="${ONLY_PLOT:-false}"

# -------- helpers --------
tagify() {
  # Convert e.g. 0.25 -> 0p25 for filesystem-friendly tags
  echo "$1" | sed 's/\./p/g'
}

fixed_outputs_exist() {
  local alpha="$1"

  # handle possible filename formatting differences: "1.0" vs "1"
  local alpha_short
  alpha_short="$(echo "$alpha" | sed 's/\.0$//')"   # 1.0 -> 1, 0.0 -> 0, 0.25 -> 0.25

  local cal1="${BASE_PLOT_PATH}/calibration_mae_alpha${alpha}.pkl"
  local val1="${BASE_PLOT_PATH}/validation_mae_alpha${alpha}.pkl"

  local cal2="${BASE_PLOT_PATH}/calibration_mae_alpha${alpha_short}.pkl"
  local val2="${BASE_PLOT_PATH}/validation_mae_alpha${alpha_short}.pkl"

  # If either (alpha) pair or (alpha_short) pair exists, skip
  if [[ -f "$cal1" && -f "$val1" ]]; then
    return 0
  fi
  if [[ -f "$cal2" && -f "$val2" ]]; then
    return 0
  fi

  return 1
}


mark_done() {
  local name="$1"
  touch "${BASE_PLOT_PATH}/.done_${name}"
}

reset_gpu() {
  # Optional: Reset GPU between runs (best-effort)
  nvidia-smi --gpu-reset 2>/dev/null || true
  sleep 5
}

run_llama_step() {
  # Build and run llama_step.py with common args
  # Usage: run_llama_step <extra args...>
  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u llama_step.py \
    "$MODEL_PATH" c4 \
    --method greedyaq \
    --sym \
    --wbits 3 \
    --true-sequential \
    --groupsize 128 \
    --seed 0 \
    --nsamples 128 \
    --eval \
    --lm-eval \
    --plot-delta-x \
    --plot-delta-x-path "${BASE_PLOT_PATH}" \
    --eval-mae-validation \
    "$@"
}

# -------- Step 1: Fixed-alpha runs (skip if already present) --------
if [ "$ONLY_PLOT" = "true" ]; then
  echo "===== [SKIP] Quantization steps (ONLY_PLOT=true) ====="
else
  echo "===== Running FIXED-alpha sweeps (skip if outputs exist) ====="
  for alpha in "${FIXED_ALPHAS[@]}"; do
  atag="$(tagify "$alpha")"
  run_id="fixed_a${atag}"

  if fixed_outputs_exist "$alpha"; then
    echo ">>> [SKIP] fixed alpha=${alpha} (outputs already exist in ${BASE_PLOT_PATH})"
    continue
  fi

  echo "===== Running FIXED alpha = ${alpha} ====="
  echo "Run id: ${run_id}"

  run_llama_step \
    --alpha-method fixed \
    --alpha "${alpha}"

  mark_done "${run_id}"
  echo "===== FIXED alpha ${alpha} completed ====="
  echo ""

  reset_gpu
  done
fi

# -------- Step 2: Sampled-alpha runs (Beta mixup) --------
if [ "$ONLY_PLOT" = "true" ]; then
  echo "===== [SKIP] Sampled-alpha runs (ONLY_PLOT=true) ====="
else
  echo "===== Running SAMPLED-alpha sweeps (Beta mixup) ====="
  for lam in "${MIXUP_PARAMS[@]}"; do
  ltag="$(tagify "$lam")"
  run_id="sample_beta${ltag}"
  marker="${BASE_PLOT_PATH}/.done_${run_id}"

  if [[ -f "$marker" ]]; then
    echo ">>> [SKIP] sampled Beta(lambda,lambda) with lambda=${lam} (marker exists: ${marker})"
    continue
  fi

  echo "===== Running SAMPLED alpha with Beta(lambda,lambda), lambda = ${lam} ====="
  echo "Run id: ${run_id}"

  run_llama_step \
    --alpha-method sample \
    --mixup-param "${lam}"

  mark_done "${run_id}"
  echo "===== Sampled mixup lambda ${lam} completed ====="
  echo ""

  reset_gpu
  done
fi

# -------- Step 3: Unified plots (calibration + validation) --------
# Only create plots if pickle files exist
echo "===== Checking for MAE pickle files ====="

# Check for validation and calibration pickle files
validation_files=$(find "${BASE_PLOT_PATH}" -name "validation_mae_*.pkl" 2>/dev/null | wc -l)
calibration_files=$(find "${BASE_PLOT_PATH}" -name "calibration_mae_*.pkl" 2>/dev/null | wc -l)

if [ "$validation_files" -eq 0 ] && [ "$calibration_files" -eq 0 ]; then
  echo ">>> [SKIP] No MAE pickle files found in ${BASE_PLOT_PATH}"
  echo ">>> Run quantization with --plot-delta-x and --eval-mae-validation first"
else
  echo "Found ${validation_files} validation MAE files and ${calibration_files} calibration MAE files"
  echo "===== Creating unified MAE plots (fixed + sampled) ====="

  CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" -u llama_step.py \
    "$MODEL_PATH" c4 \
    --plot-unified-mae "${BASE_PLOT_PATH}"

  echo "===== All done! Check plots:"
  echo "  - Validation MAE: ${BASE_PLOT_PATH}/unified_validation_mae.png"
  echo "  - Calibration MAE: ${BASE_PLOT_PATH}/unified_calibration_mae.png"
fi
