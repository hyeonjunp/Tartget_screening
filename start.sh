#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

APP_ENV="$ROOT_DIR/.target-screening-envs/app"
VINA_ENV="$ROOT_DIR/.target-screening-envs/vina"
RXNFLOW_ENV="$ROOT_DIR/RxnFlow/.conda"
RXNFLOW_DATA="$ROOT_DIR/RxnFlow/data/envs/zincfrag"
BUILD_CPUS="${BUILD_CPUS:-8}"
PORT="${STREAMLIT_SERVER_PORT:-8501}"

if [[ -f "$ROOT_DIR/.env" ]]; then
  set -a
  source "$ROOT_DIR/.env"
  set +a
fi

say() { printf '\n==> %s\n' "$*"; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

find_env_manager() {
  if [[ -n "${CONDA_EXE:-}" && -x "$CONDA_EXE" ]]; then
    printf '%s' "$CONDA_EXE"
  elif command -v conda >/dev/null 2>&1; then
    command -v conda
  elif command -v micromamba >/dev/null 2>&1; then
    command -v micromamba
  elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    printf '%s' "$HOME/miniconda3/bin/conda"
  elif [[ -x "$HOME/anaconda3/bin/conda" ]]; then
    printf '%s' "$HOME/anaconda3/bin/conda"
  else
    printf ''
  fi
}

ENV_MANAGER="$(find_env_manager)"
if [[ -z "$ENV_MANAGER" ]]; then
  command -v curl >/dev/null 2>&1 || fail "curl is required to install micromamba."
  command -v tar >/dev/null 2>&1 || fail "tar is required to install micromamba."
  say "Conda was not found; downloading micromamba"
  mkdir -p "$ROOT_DIR/.tools"
  curl -fsSL https://micro.mamba.pm/api/micromamba/linux-64/latest |
    tar -xj -C "$ROOT_DIR/.tools" --strip-components=1 bin/micromamba
  ENV_MANAGER="$ROOT_DIR/.tools/micromamba"
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$ROOT_DIR/.mamba}"

if [[ ! -x "$APP_ENV/bin/streamlit" ]]; then
  mkdir -p "$(dirname "$APP_ENV")"
  if [[ -x "$APP_ENV/bin/python" ]]; then
    say "Completing the app environment"
    "$ENV_MANAGER" env update -y -p "$APP_ENV" -f "$ROOT_DIR/environment.yml"
  else
    say "Installing the app environment (Python 3.12, first launch can take a while)"
    "$ENV_MANAGER" env create -y -p "$APP_ENV" -f "$ROOT_DIR/environment.yml"
  fi
fi

if [[ ! -x "$VINA_ENV/bin/python" ]]; then
  say "Creating the AutoDock Vina environment"
  "$ENV_MANAGER" create -y -p "$VINA_ENV" python=3.12 pip
fi
if ! "$VINA_ENV/bin/python" -c "import vina, meeko, gemmi, rdkit" >/dev/null 2>&1 ||
   [[ ! -x "$VINA_ENV/bin/mk_prepare_receptor.py" ]]; then
  say "Installing AutoDock Vina and receptor preparation tools"
  "$VINA_ENV/bin/python" -m pip install --upgrade pip
  "$VINA_ENV/bin/python" -m pip install vina meeko gemmi rdkit
fi

if [[ ! -x "$RXNFLOW_ENV/bin/python" ]]; then
  say "Creating the RxnFlow Python 3.12 environment"
  "$ENV_MANAGER" create -y -p "$RXNFLOW_ENV" python=3.12 pip
fi
if ! "$RXNFLOW_ENV/bin/python" -c "import torch, torch_geometric, pmnet_appl, rxnflow, rdkit, tqdm, gdown" >/dev/null 2>&1; then
  say "Installing RxnFlow and its CPU dependencies"
  (
    cd "$ROOT_DIR/RxnFlow"
    "$RXNFLOW_ENV/bin/python" -m pip install --upgrade pip
    "$RXNFLOW_ENV/bin/python" -m pip install \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --find-links https://data.pyg.org/whl/torch-2.5.1+cpu.html \
      torch==2.5.1 torch-scatter torch-sparse torch-cluster \
      ".[pmnet]" gdown tqdm
  )
fi

required=(template.txt building_block.smi bb_mask.npy bb_fp_2_1024.npy bb_desc.npy)
data_ready=true
for file in "${required[@]}"; do
  [[ -s "$RXNFLOW_DATA/$file" ]] || data_ready=false
done
if [[ "$data_ready" != true ]]; then
  say "Preparing RxnFlow ZINCFrag building-block data"
  mkdir -p "$ROOT_DIR/RxnFlow/data/building_blocks"
  BLOCKS="$ROOT_DIR/RxnFlow/data/building_blocks/zincfrag.smi.gz"
  if [[ ! -s "$BLOCKS" ]]; then
    "$RXNFLOW_ENV/bin/gdown" \
      "https://drive.google.com/uc?id=16N8Xyxr9a-CifjIofgdH3ssFukC4Eh_V" \
      -O "$BLOCKS"
  fi
  mkdir -p "$ROOT_DIR/RxnFlow/data/envs"
  TMP_DATA="$ROOT_DIR/RxnFlow/data/envs/zincfrag.setup.$$"
  (
    cd "$ROOT_DIR/RxnFlow"
    "$RXNFLOW_ENV/bin/python" data/scripts/b_create_env.py \
      -b "$BLOCKS" -o "$TMP_DATA" -t data/templates/real.txt --cpu "$BUILD_CPUS"
  )
  if [[ -e "$RXNFLOW_DATA" ]]; then
    mv "$RXNFLOW_DATA" "$RXNFLOW_DATA.incomplete.$(date +%Y%m%d%H%M%S)"
  fi
  mv "$TMP_DATA" "$RXNFLOW_DATA"
fi

export VINA_PYTHON="$VINA_ENV/bin/python"
export RXNFLOW_PYTHON="$RXNFLOW_ENV/bin/python"
export RXNFLOW_ENV_DIR="$RXNFLOW_DATA"

say "Starting Target Screening at http://localhost:$PORT"
exec "$APP_ENV/bin/streamlit" run "$ROOT_DIR/app/ui.py" \
  --server.port "$PORT" \
  --server.address "${STREAMLIT_SERVER_ADDRESS:-127.0.0.1}" \
  --server.headless "${STREAMLIT_SERVER_HEADLESS:-false}"
