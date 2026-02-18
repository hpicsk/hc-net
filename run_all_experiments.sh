#!/bin/bash
# NeurIPS HC-Net: Full Experiment Suite
# Run all 7 experiments with full parameters.
# Estimated total runtime: ~30-55 hours (dominated by exp4).

set -e
export CUDA_VISIBLE_DEVICES=1

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="./results/logs_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "HC-Net NeurIPS Full Experiment Suite"
echo "Started: $(date)"
echo "GPU: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Logs: $LOG_DIR"
echo "=============================================="

# --- exp1: Spinning System 2D (~15min) ---
echo ""
echo "[1/7] Experiment 1: Spinning System 2D"
python -m nips_hcnet.train --experiment exp1 \
    --seeds 42 123 456 --n_epochs 100 \
    2>&1 | tee "$LOG_DIR/exp1.log"

# --- exp2: Chirality Grade Hierarchy (~45min) ---
echo ""
echo "[2/7] Experiment 2: Chirality Grade Hierarchy"
for MODE in rotation chirality spiral; do
    echo "  Mode: $MODE"
    python -m nips_hcnet.train --experiment exp2 \
        --mode $MODE --seeds 42 123 456 --n_epochs 100 \
        2>&1 | tee "$LOG_DIR/exp2_${MODE}.log"
done

# --- exp3: 3D N-Body Chirality (~30min) ---
echo ""
echo "[3/7] Experiment 3: 3D N-Body Chirality"
for MODE in chirality rotation; do
    echo "  Mode: $MODE"
    python -m nips_hcnet.train --experiment exp3 \
        --mode $MODE --seeds 42 123 456 --n_epochs 100 \
        2>&1 | tee "$LOG_DIR/exp3_${MODE}.log"
done

# --- exp4: MD17 Force Prediction — CRITICAL (~24-48h) ---
echo ""
echo "[4/7] Experiment 4: MD17 Force Prediction (energy-conserving)"
python -m nips_hcnet.train --experiment exp4 \
    --all_molecules --models hybrid_hcnet_energy egnn \
    --seeds 42 123 456 --n_train 9500 --n_epochs 100 \
    2>&1 | tee "$LOG_DIR/exp4.log"

# --- exp5: Scaling Analysis (~20min) ---
echo ""
echo "[5/7] Experiment 5: Scaling Analysis"
python -m nips_hcnet.train --experiment exp5 \
    2>&1 | tee "$LOG_DIR/exp5.log"

# --- exp6: 3D SOTA Comparison (~4-8h) ---
echo ""
echo "[6/7] Experiment 6: 3D SOTA Comparison"
python -m nips_hcnet.train --experiment exp6 \
    --seeds 42 123 456 --epochs 100 \
    2>&1 | tee "$LOG_DIR/exp6.log"

# --- exp7: Ablation Study (~2-4h) ---
echo ""
echo "[7/7] Experiment 7: Ablation Study"
python -m nips_hcnet.train --experiment exp7 \
    --seeds 42 123 456 --epochs 100 \
    2>&1 | tee "$LOG_DIR/exp7.log"

echo ""
echo "=============================================="
echo "All experiments completed: $(date)"
echo "Logs saved to: $LOG_DIR"
echo "=============================================="
