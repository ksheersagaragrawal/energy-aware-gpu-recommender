#!/usr/bin/env bash

LOG_DIR="results/reports"
LOG_FILE="${LOG_DIR}/phase4_download_log.txt"

mkdir -p "external" "data/raw/papers" "data/raw/external_benchmarks" "$LOG_DIR"

echo "[phase4] download started at $(date)" > "$LOG_FILE"

log() {
  echo "[phase4] $1" | tee -a "$LOG_FILE"
}

try_download() {
  local url="$1"
  local out="$2"
  if [ -z "$url" ]; then
    log "missing url for $out"
    return 0
  fi
  if [ -f "$out" ]; then
    log "already exists: $out"
    return 0
  fi
  if curl -L --fail -o "$out" "$url"; then
    log "downloaded $url"
  else
    log "failed to download $url"
  fi
}

try_clone() {
  local repo="$1"
  local dest="$2"
  if [ -d "$dest" ]; then
    log "already cloned: $dest"
    return 0
  fi
  if git clone "$repo" "$dest"; then
    log "cloned $repo"
  else
    log "failed to clone $repo"
  fi
}

# PDFs (URLs unknown in this script; add if available)
try_download "" "data/raw/papers/wu2015gpgpu.pdf"
try_download "" "data/raw/papers/dutta2018gpu_power.pdf"
try_download "" "data/raw/papers/moolchandani2022concurrent.pdf"
try_download "" "data/raw/papers/braun2021_mangrove.pdf"
try_download "" "data/raw/papers/mlenergy2025.pdf"

# Repos
try_clone "https://github.com/lorenzbraun/gpu-mangrove" "external/gpu-mangrove"
try_clone "https://github.com/ml-energy/benchmark" "external/ml-energy-benchmark"

log "download finished"
