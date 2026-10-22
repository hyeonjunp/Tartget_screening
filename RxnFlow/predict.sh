#!/bin/bash

# ==============================================================================
# RxnFlow Proxy Prediction Script
# ==============================================================================
# This script predicts the Proxy (Vina) scores for molecules provided in an SDF,
# TXT, or CSV file against a target protein binding pocket.
# ==============================================================================

TARGET="6S89"
INPUT_MOL="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/molecule/6s89_B_L0Q.sdf"
PROTEIN="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/target/${TARGET}.pdb"
REF_LIGAND="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/molecule/6s89_B_L0Q.sdf"
OUT_PATH="./results/${TARGET}/baseline.csv"

# Add src to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "---------------------------------------------------------"
echo "Starting Proxy & UniDock Prediction for $TARGET..."
echo "Input Molecules: $INPUT_MOL"
echo "Target Protein : $PROTEIN"
echo "Output Results : $OUT_PATH"
echo "---------------------------------------------------------"

python scripts/predict.py \
    --input_file "$INPUT_MOL" \
    --out_path "$OUT_PATH" \
    --protein "$PROTEIN" \
    --ref_ligand "$REF_LIGAND" \
    --cuda \
    --batch_size 10