#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  —  M4 XAI Benchmark Replication (NeurIPS 2023)
# =============================================================================
# Replicates Table 1: ResNet-50, ImageNet-val 5000 images, 5 attribution
# methods × 3 metrics (MoRF, LeRF, ABPC).
#
# All settings are hardcoded to match the paper exactly. Just run:
#   bash run_experiments.sh
# =============================================================================

set -euo pipefail

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERR]${NC}   $*" >&2; }
banner()  { echo -e "\n${BOLD}${CYAN}━━━  $*  ━━━${NC}\n"; }

# =============================================================================
# ██  HARDCODED SETTINGS  (edit here to change anything)
# =============================================================================

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
MAIN_SCRIPT="$SCRIPT_DIR/table1_resnet50.py"
DATA_DIR="$SCRIPT_DIR/data/M4_5000"          # 1000 classes × 5 images
RESULTS_ROOT="$SCRIPT_DIR/results"
LOG_DIR="$RESULTS_ROOT/logs"

# ── Attribution methods ───────────────────────────────────────────────────────
METHODS="constant random_16 random gradcam ig sg"   # all six from Table 1

# ── Perturbation (MoRF / LeRF / ABPC) ────────────────────────────────────────
N_STEPS=20          # paper: 20 percentile steps (each covers 5% of pixels)

# ── Integrated Gradients ──────────────────────────────────────────────────────
IG_STEPS=50         # number of Riemann-sum steps along the baseline path

# ── SmoothGrad ────────────────────────────────────────────────────────────────
SG_SAMPLES=50       # number of noisy forward passes
SG_STDEV=0.15       # noise stdev = 0.15 × (input_max − input_min)

# ── Smoke test (sanity check before the full run) ─────────────────────────────
SMOKE_IMAGES=100    # 20 classes × 5 images
SMOKE_IG_STEPS=10   # faster IG for the smoke test
SMOKE_SG_SAMPLES=10 # faster SG for the smoke test

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED=42

# =============================================================================

mkdir -p "$LOG_DIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
SMOKE_LOG="$LOG_DIR/smoke_${RUN_ID}.log"
FULL_LOG="$LOG_DIR/full_${RUN_ID}.log"

banner "M4 XAI Benchmark — Table 1 Replication"
info "Run ID      : $RUN_ID"
info "Data        : $DATA_DIR"
info "Methods     : $METHODS"
info "n_steps     : $N_STEPS   (paper: 20)"
info "ig_steps    : $IG_STEPS  (paper: ~50)"
info "sg_samples  : $SG_SAMPLES (paper: ~50)"
info "sg_stdev    : $SG_STDEV"
info "seed        : $SEED"
info "Results dir : $RESULTS_ROOT"
echo

# ── Pre-flight checks ─────────────────────────────────────────────────────────
banner "Pre-flight Checks"

[[ -f "$PYTHON" ]]      || { error "venv not found at $VENV. Run: uv sync"; exit 1; }
[[ -f "$MAIN_SCRIPT" ]] || { error "Script not found: $MAIN_SCRIPT";         exit 1; }
[[ -d "$DATA_DIR" ]]    || { error "Dataset not found: $DATA_DIR";            exit 1; }

N_IMAGES="$(find "$DATA_DIR" -name '*.JPEG' | wc -l)"
N_CLASSES="$(ls "$DATA_DIR" | wc -l)"
success "Dataset   : $N_IMAGES images across $N_CLASSES classes"

GPU_INFO="$("$PYTHON" -c "import torch; print('cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu (no CUDA)')" 2>/dev/null)"
success "Device    : $GPU_INFO"

TORCH_VER="$("$PYTHON" -c "import torch; print(torch.__version__)" 2>/dev/null)"
success "PyTorch   : $TORCH_VER"

"$PYTHON" -c "import captum" 2>/dev/null \
    && success "captum    : installed" \
    || { error "captum not installed. Run: uv pip install captum"; exit 1; }

"$PYTHON" -c "import pytorch_grad_cam" 2>/dev/null \
    && success "grad-cam  : installed" \
    || warn "grad-cam  : NOT installed — GradCAM falls back to gradient×input"

# ── Stage 1: Smoke test ───────────────────────────────────────────────────────
banner "Stage 1 — Smoke Test (${SMOKE_IMAGES} images / 20 classes)"
info "Log: $SMOKE_LOG"

SMOKE_DIR="$RESULTS_ROOT/smoke_${RUN_ID}"

"$PYTHON" "$MAIN_SCRIPT" \
    --data_dir    "$DATA_DIR" \
    --methods     $METHODS \
    --n_steps     $N_STEPS \
    --ig_steps    $SMOKE_IG_STEPS \
    --sg_samples  $SMOKE_SG_SAMPLES \
    --sg_stdev    $SG_STDEV \
    --output_dir  "$SMOKE_DIR" \
    --seed        $SEED \
    --limit       $SMOKE_IMAGES \
    2>&1 | tee "$SMOKE_LOG"

success "Smoke test passed ✓"

# ── Stage 2: Full evaluation ──────────────────────────────────────────────────
banner "Stage 2 — Full Evaluation (all $N_IMAGES images)"

FULL_DIR="$RESULTS_ROOT/table1_${RUN_ID}"

info "Output dir : $FULL_DIR"
info "Log        : $FULL_LOG"
info "Estimated time on a single GPU:"
info "  constant + random : ~2 min"
info "  gradcam           : ~20 min"
info "  ig (50 steps)     : ~90 min"
info "  sg (50 samples)   : ~90 min"
info "  ─────────────────────────────"
info "  Total             : ~3.5 hours"
echo

START_TS=$(date +%s)

"$PYTHON" "$MAIN_SCRIPT" \
    --data_dir    "$DATA_DIR" \
    --methods     $METHODS \
    --n_steps     $N_STEPS \
    --ig_steps    $IG_STEPS \
    --sg_samples  $SG_SAMPLES \
    --sg_stdev    $SG_STDEV \
    --output_dir  "$FULL_DIR" \
    --seed        $SEED \
    2>&1 | tee "$FULL_LOG"

END_TS=$(date +%s)
ELAPSED=$(( END_TS - START_TS ))
MINS=$(( ELAPSED / 60 ))
SECS=$(( ELAPSED % 60 ))

success "Full evaluation complete in ${MINS}m ${SECS}s ✓"
success "Results → $FULL_DIR/table1_resnet50.csv"

# ── Stage 3: Compare against paper Table 1 ───────────────────────────────────
banner "Stage 3 — Compare Against Paper Table 1 (ResNet-50)"

CSV="$FULL_DIR/table1_resnet50.csv"

if [[ -f "$CSV" ]]; then
    "$PYTHON" - "$CSV" <<'PYEOF'
import csv, sys

csv_path = sys.argv[1]

# Paper Table 1 reference values — ResNet-50 (NeurIPS 2023)
paper = {
    "constant":  dict(MoRF=0.000, ABPC=0.000, INFD=3.015),
    "random_16": dict(MoRF=0.596, ABPC=0.007, INFD=3.039),
    "random":    dict(MoRF=0.599, ABPC=0.008, INFD=3.015),
    "gradcam":   dict(MoRF=0.628, ABPC=0.424, INFD=2.496),
    "ig":        dict(MoRF=0.709, ABPC=0.377, INFD=2.373),
    "sg":        dict(MoRF=0.701, ABPC=0.369, INFD=2.323),
}

print(f"\n  Loaded: {csv_path}\n")
print(f"  {'Method':<12}  {'Metric':<6}  {'Ours':>8}  {'Paper':>8}  {'Δ':>8}  {'Match'}")
print("  " + "-" * 58)

with open(csv_path) as f:
    rows = list(csv.DictReader(f))

for row in rows:
    m = row["method"]
    if m not in paper:
        continue
    for metric in ["MoRF", "ABPC", "INFD"]:
        val = row.get(metric, "nan")
        if val in ("nan", "", "N/A"):
            continue
        ours  = float(val)
        ref   = paper[m].get(metric)
        if ref is None:
            continue
        delta = ours - ref
        flag  = "✓" if abs(delta) < 0.05 else ("△" if abs(delta) < 0.10 else "✗")
        print(f"  {m:<12}  {metric:<6}  {ours:>8.4f}  {ref:>8.4f}  {delta:>+8.4f}  {flag}")

print()
print("  Legend: ✓ Δ<0.05  △ Δ<0.10  ✗ Δ≥0.10")
PYEOF
else
    warn "CSV not found — skipping comparison."
fi

# ── Done ──────────────────────────────────────────────────────────────────────
banner "All Done"
echo -e "${GREEN}Logs    :${NC} $LOG_DIR/"
echo -e "${GREEN}Results :${NC} $FULL_DIR/"
echo
