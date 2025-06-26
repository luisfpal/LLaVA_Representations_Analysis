#!/bin/bash
# ====================== Configuration ====================== #
ACCOUNT="dssc"
JOB_NAME="experiment"
PARTITION="DGX"
GPUS=1
# Extract heads representations parameters
# CPUS=40
# MEM="70GB"
# TIME="0:30:00"
# Less demanding parameters
CPUS=20
MEM="20GB"
TIME="00:10:00"

PROJECT_ROOT="$HOME/multimodal_finetuned_representations"
LOG_DIR="$PROJECT_ROOT/.experiments_logs"
mkdir -p "$LOG_DIR"

# SCRIPT="evaluate_benchmarks.sh"
# SCRIPT="extract_residual_streams.sh"
# SCRIPT="analyze_residual_streams.sh"
# SCRIPT="intrinsic_dimension_estimation.sh"
SCRIPT="renyi_entropy_evaluation.sh"
# SCRIPT="extract_heads_representations.sh"
# SCRIPT="analyze_heads_representations.sh"

# ====================== S  BATCH Script Creation ====================== #
SBATCH_SCRIPT=$(mktemp /tmp/sbatch_experiment.XXXXXX.sh)

cat <<EOT >"$SBATCH_SCRIPT"
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --job-name=$JOB_NAME
#SBATCH --partition=$PARTITION
#SBATCH --gres=gpu:$GPUS
#SBATCH --cpus-per-task=$CPUS
#SBATCH --mem=$MEM
#SBATCH --time=$TIME
#SBATCH --output=$LOG_DIR/slurm-%j.out
#SBATCH --error=$LOG_DIR/slurm-%j.out

# Activate environment
source "$PROJECT_ROOT/.venv/bin/activate"

# Run script
bash "$PROJECT_ROOT/shell/$SCRIPT"
EOT

# ====================== Job Submission ====================== #
echo "🟢 Submitting SBATCH job for experiment: $SCRIPT"
echo "----------------------------------------------------"
cat "$SBATCH_SCRIPT"
echo "----------------------------------------------------"

JOB_ID=$(sbatch "$SBATCH_SCRIPT" | awk '{print $4}')
echo "✅ Job submitted with ID: $JOB_ID"
echo "📁 Unified output: $LOG_DIR/slurm-$JOB_ID.out"
