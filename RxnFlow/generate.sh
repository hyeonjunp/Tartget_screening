#!/bin/bash

export PYTHONPATH=$PYTHONPATH:$(pwd)/src

MODEL="qvina-unif-0-64"
ENV_DIR="./data/envs/catalog"
NUM_SAMPLES=100


OUTPUT_BASE="./test/6S89"
mkdir -p ${OUTPUT_BASE}

# ---------------------------------------------------------
# Target 1: 6S89
# ---------------------------------------------------------
P1_PROTEIN="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/target/6S89.pdb"
P1_LIGAND="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/molecule/6s89_B_L0Q.sdf"

P1_OUT="${OUTPUT_BASE}/smiles.csv"

echo "Starting sampling for 6S89 (Zero-shot, No UniDock)..."
python scripts/sampling_zeroshot.py \
  --model_path $MODEL \
  --env_dir $ENV_DIR \
  -p $P1_PROTEIN \
  -l $P1_LIGAND \
  -o $P1_OUT \
  -n $NUM_SAMPLES \
  --cuda

OUTPUT_BASE="./test/6S8A"
mkdir -p ${OUTPUT_BASE}

# ---------------------------------------------------------
# Target 2: 6S8A
# ---------------------------------------------------------
P2_PROTEIN="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/target/6S8A.pdb"
P2_LIGAND="/home/rlawlsgurjh/hdd/work/gitlab/MGM/dataset/molecule/6s8a_C_L0N.sdf"
P2_OUT="${OUTPUT_BASE}/smiles.csv"

echo "Starting sampling for 6S8A (Zero-shot, No UniDock)..."
python scripts/sampling_zeroshot.py \
  --model_path $MODEL \
  --env_dir $ENV_DIR \
  -p $P2_PROTEIN \
  -l $P2_LIGAND \
  -o $P2_OUT \
  -n $NUM_SAMPLES \
  --cuda

echo "All sampling tasks completed."
