#!/bin/bash

MODEL="qvina-unif-0-64"
ENV_DIR="./data/envs/zincfrag" 
NUM_SAMPLES=5000

TRIAL="Trial_2"

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

INPUT_BASE="/home/rlawlsgurjh/hdd/work/gitlab/MGM/RxnFlow"

OUTPUT_BASE="./results/6S89/${TRIAL}"
mkdir -p ${OUTPUT_BASE}

# ---------------------------------------------------------
# Target 1: 6S89
# ---------------------------------------------------------
P1_PROTEIN="${INPUT_BASE}/../dataset/target/6S89.pdb"
P1_LIGAND="${INPUT_BASE}/../dataset/molecule/6s89_B_L0Q.sdf"
P1_OUT="${OUTPUT_BASE}/smiles.csv"

echo "Starting sampling for 6S89 (Valid & Unique, No UniDock)..."
python scripts/custom_sampling_zeroshot.py \
  --model_path $MODEL \
  --env_dir $ENV_DIR \
  -p $P1_PROTEIN \
  -l $P1_LIGAND \
  -o $P1_OUT \
  -n $NUM_SAMPLES \
  --batch_size 100 \
  --cuda

# OUTPUT_BASE="./results/6S8A/${TRIAL}"
# mkdir -p ${OUTPUT_BASE}

# # ---------------------------------------------------------
# # Target 2: 6S8A
# # ---------------------------------------------------------
# P2_PROTEIN="${INPUT_BASE}/../dataset/target/6S8A.pdb"
# P2_LIGAND="${INPUT_BASE}/../dataset/molecule/6s8a_C_L0N.sdf"
# P2_OUT="${OUTPUT_BASE}/smiles.csv"

# echo "Starting sampling for 6S8A (Valid & Unique, No UniDock)..."
# python scripts/custom_sampling_zeroshot.py \
#   --model_path $MODEL \
#   --env_dir $ENV_DIR \
#   -p $P2_PROTEIN \
#   -l $P2_LIGAND \
#   -o $P2_OUT \
#   -n $NUM_SAMPLES \
#   --batch_size 100 \
#   --cuda

# echo "All sampling tasks completed."
