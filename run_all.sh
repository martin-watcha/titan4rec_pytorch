#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Titan4Rec 전체 실험 스크립트
#
# Usage:
#   bash run_all.sh [DATASET...]
#
#   # 전체 실험 (기본: ml-100k ml-1m)
#   bash run_all.sh
#
#   # 특정 데이터셋
#   bash run_all.sh ml-100k
#   bash run_all.sh ml-100k ml-1m ml-10m
#
# 환경변수로 제어:
#   PHASE=all|baseline|ablation   (기본: all)
#   MODELS="sasrec bert4rec ..."  (기본: 4개 전체)
#   BS=128                        (baseline batch size, 기본: 128)
#   BS_TITAN=128                  (titan4rec batch size, 기본: 128)
#   EPOCHS=200                    (최대 epoch, 기본: 200)
#   PATIENCE=10                   (early stopping, 기본: 10)
#   USE_WANDB=1                   (W&B 로깅 활성화)
#   WANDB_PROJECT=titan4rec       (W&B 프로젝트명)
#
# 예시:
#   PHASE=baseline bash run_all.sh ml-100k
#   PHASE=ablation PATIENCE=5 bash run_all.sh ml-1m
#   MODELS="sasrec titan4rec" bash run_all.sh ml-100k ml-1m
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")"

# ── 설정 ────────────────────────────────────────────────────────────────────
PHASE="${PHASE:-all}"
BS="${BS:-128}"
BS_TITAN="${BS_TITAN:-128}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-10}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-titan4rec}"

if [ -z "${MODELS:-}" ]; then
    BASELINE_MODELS=(sasrec bert4rec mamba4rec titan4rec)
else
    read -ra BASELINE_MODELS <<< "$MODELS"
fi

if [ $# -eq 0 ]; then
    DATASETS=(ml-100k ml-1m)
else
    DATASETS=("$@")
fi

# W&B 플래그
WANDB_FLAG=""
[ "$USE_WANDB" = "1" ] && WANDB_FLAG="--use_wandb --wandb_project ${WANDB_PROJECT}"

# 로그 디렉토리
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

# ── 헬퍼 ────────────────────────────────────────────────────────────────────
START_TIME=$(date +%s)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_exp() {
    local label="$1"; shift
    local logfile="${LOG_DIR}/${label//[^a-zA-Z0-9_-]/_}.log"
    log "START  $label"
    python train.py "$@" 2>&1 | tee "$logfile" | grep --line-buffered "^\[Epoch" || true
    local py_exit
    py_exit="${PIPESTATUS[0]}"
    if [ "$py_exit" -eq 0 ]; then
        local ndcg
        ndcg=$(grep -oP 'val_ndcg@10=\K[0-9.]+' "$logfile" | tail -1)
        log "DONE   $label  → val NDCG@10=${ndcg:-?}"
    else
        log "FAILED $label (log: $logfile)"
    fi
}

elapsed() {
    local s=$(( $(date +%s) - START_TIME ))
    printf "%02d:%02d:%02d" $((s/3600)) $((s%3600/60)) $((s%60))
}

# ── 실험 ────────────────────────────────────────────────────────────────────
log "============================================================"
log " Titan4Rec Experiments"
log " Datasets : ${DATASETS[*]}"
log " Phase    : $PHASE"
log " Epochs   : $EPOCHS  Patience: $PATIENCE"
log " Log dir  : $LOG_DIR/"
log "============================================================"

for DATASET in "${DATASETS[@]}"; do
    log ""
    log "══════════════════ $DATASET ══════════════════"

    # ── Phase 1: Baseline Comparison ────────────────────────────────────────
    if [[ "$PHASE" == "all" || "$PHASE" == "baseline" ]]; then
        log "── Baseline Comparison ──"
        for MODEL in "${BASELINE_MODELS[@]}"; do
            BS_CUR=$BS
            [ "$MODEL" = "titan4rec" ] && BS_CUR=$BS_TITAN
            run_exp "${DATASET}_${MODEL}" \
                --model_name "$MODEL" \
                --dataset "$DATASET" \
                --batch_size "$BS_CUR" \
                --num_epochs "$EPOCHS" \
                --patience "$PATIENCE" \
                $WANDB_FLAG
        done
    fi

    # ── Phase 2: Ablation Study ─────────────────────────────────────────────
    if [[ "$PHASE" == "all" || "$PHASE" == "ablation" ]]; then
        log "── Ablation Study (titan4rec) ──"
        COMMON=(--model_name titan4rec --dataset "$DATASET"
                --batch_size "$BS_TITAN" --num_epochs "$EPOCHS"
                --patience "$PATIENCE" $WANDB_FLAG)

        # Memory depth
        for DEPTH in 1 3; do
            run_exp "${DATASET}_titan4rec_mem_depth${DEPTH}" \
                "${COMMON[@]}" --memory_depth "$DEPTH"
        done

        # Persistent memory
        run_exp "${DATASET}_titan4rec_persistent0" \
            "${COMMON[@]}" --num_persistent 0

        # Segment size
        for SEG in 10 50; do
            run_exp "${DATASET}_titan4rec_seg${SEG}" \
                "${COMMON[@]}" --segment_size "$SEG"
        done

        # TBPTT window
        for K in 1 5; do
            run_exp "${DATASET}_titan4rec_tbptt${K}" \
                "${COMMON[@]}" --tbptt_k "$K"
        done
    fi

    log "── $DATASET complete (elapsed: $(elapsed)) ──"
done

# ── 결과 요약 ────────────────────────────────────────────────────────────────
log ""
log "============================================================"
log " All experiments complete  (total: $(elapsed))"
log " Results:"
for DATASET in "${DATASETS[@]}"; do
    CSV="results/${DATASET}_results.csv"
    if [ -f "$CSV" ]; then
        log "   $CSV"
        # 최근 실험 결과 상위 5개 출력
        tail -n +2 "$CSV" | sort -t',' -k6 -rn | head -5 | \
            awk -F',' '{printf "     %-40s val=%s  test=%s\n", $1, $6, $8}'
    fi
done
log "============================================================"
