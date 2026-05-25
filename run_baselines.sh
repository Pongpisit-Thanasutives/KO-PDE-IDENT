#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
#  run_baselines.sh
#  ────────────────
#  Runs STLSQ, SR3, SSR baseline comparisons on all five datasets.
#  Results are saved to individual log files and a combined summary.
#
#  Usage:
#    chmod +x run_baselines.sh
#    ./run_baselines.sh
#
#  Configuration:
#    - Set DATADIR to the directory containing your .mat files.
#    - Set SCRIPT to the path to baseline_comparison.py.
#    - Set OUTDIR to where log files should be written.
# ═══════════════════════════════════════════════════════════════════

set -e
set -o pipefail

# ── Configuration ────────────────────────────────────────────────
DATADIR="${DATADIR:-../Datasets}"
SCRIPT="${SCRIPT:-baseline_comparison.py}"
OUTDIR="${OUTDIR:-$DATADIR/baseline_results}"
CV=3
METHODS="STLSQ SR3 SSR"

# ── Dataset definitions ──────────────────────────────────────────
# Format: "short_name|filename"
DATASETS=(
    "burgers|burgers_weak_form_noise50_sample2000.mat"
    "kdv|KdV_weak_form_noise50_sample2000.mat"
    "ks|KS_weak_form_noise50_sample2000.mat"
    "rd|RD_weak_form_noise10_sample10000.mat"
    "gs|GS_weak_form_noise0-1_sample10000.mat"
)

# ── Setup ────────────────────────────────────────────────────────
mkdir -p "$OUTDIR"
SUMMARY="$OUTDIR/summary.txt"
> "$SUMMARY"

echo "═══════════════════════════════════════════════════════════════"
echo "  Baseline Comparison: STLSQ / SR3 / SSR"
echo "  Data directory: $DATADIR"
echo "  Output directory: $OUTDIR"
echo "  CV folds: $CV"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# ── Run each dataset ─────────────────────────────────────────────
PASS=0
FAIL=0

for entry in "${DATASETS[@]}"; do
    IFS="|" read -r name filename <<< "$entry"
    datapath="$DATADIR/$filename"
    logfile="$OUTDIR/${name}.log"

    echo "──────────────────────────────────────────────────────────"
    echo "  [$name] $filename"
    echo "──────────────────────────────────────────────────────────"

    if [ ! -f "$datapath" ]; then
        echo "  SKIPPED: $datapath not found"
        echo "[$name] SKIPPED: file not found" >> "$SUMMARY"
        echo ""
        continue
    fi

    # Run and tee to both terminal and log
    if python3 "$SCRIPT" --data "$datapath" --methods $METHODS --cv "$CV" \
        2>&1 | tee "$logfile"; then
        PASS=$((PASS + 1))
        echo "[$name] OK — see $logfile" >> "$SUMMARY"
    else
        FAIL=$((FAIL + 1))
        echo "[$name] FAILED — see $logfile" >> "$SUMMARY"
    fi

    # Extract summary table to combined file
    echo "" >> "$SUMMARY"
    echo "=== $name ===" >> "$SUMMARY"
    sed -n '/^=====/,/^=====/p' "$logfile" >> "$SUMMARY"
    echo "" >> "$SUMMARY"

    # Extract LaTeX rows
    grep '^ &' "$logfile" >> "$OUTDIR/latex_rows.txt" 2>/dev/null || true

    echo ""
done

# ── Final summary ────────────────────────────────────────────────
echo "═══════════════════════════════════════════════════════════════"
echo "  Done: $PASS succeeded, $FAIL failed, $((${#DATASETS[@]} - PASS - FAIL)) skipped"
echo "  Logs:   $OUTDIR/*.log"
echo "  Summary: $SUMMARY"
echo "  LaTeX:   $OUTDIR/latex_rows.txt"
echo "═══════════════════════════════════════════════════════════════"
