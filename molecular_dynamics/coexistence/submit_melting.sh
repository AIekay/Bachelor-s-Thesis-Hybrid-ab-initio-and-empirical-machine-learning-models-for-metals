#!/bin/bash
# Submit all melting-point interface MD jobs, then a dependent calc job.
# Run from this directory: bash submit_melting.sh

MELT_TEMPS=(1200 1300 1400)

echo "=== Submitting MELTING jobs ==="
JOB_IDS=()
for T in "${MELT_TEMPS[@]}"; do
    JID=$(sbatch --job-name="melt_${T}K" job_melting.sh $T | awk '{print $NF}')
    echo "  melting ${T} K -> job $JID"
    JOB_IDS+=($JID)
done

# Build afterok dependency string: afterok:id1:id2:id3
DEP=$(IFS=:; echo "afterok:${JOB_IDS[*]}")

echo ""
echo "=== Submitting CALC job (runs after all MD jobs finish) ==="
CALC_JID=$(sbatch --dependency=$DEP --job-name="calc_melt" \
    job_calc_melting.sh ${MELT_TEMPS[*]} | awk '{print $NF}')
echo "  calc_melting -> job $CALC_JID (depends on ${JOB_IDS[*]})"

echo ""
echo "Check queue with: squeue -u \$USER"
