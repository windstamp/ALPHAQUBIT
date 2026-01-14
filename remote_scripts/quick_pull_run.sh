#!/bin/bash
# =============================================================================
# quick_pull_run.sh - Quick one-liner to pull and run pipeline
# =============================================================================
# Usage on remote server:
#   curl -sL https://raw.githubusercontent.com/xuda1979/ALPHAQUBIT/main/remote_scripts/quick_pull_run.sh | bash
# Or:
#   wget -qO- [S3_URL] | bash
# =============================================================================

S3_REMOTE="nm-aihuanxin:jtdlp-3ed7854b946a47b1a49ad754baa76cd3"
WORK_DIR="${HOME}/work/ALPHAQUBIT"
GOOGLE_DATA="${HOME}/work/google_qec3v5_experiment_data"

echo "=== Quick Pull & Run Pipeline ==="

# Pull essential files
mkdir -p ${WORK_DIR}
cd ${WORK_DIR}

echo "Pulling from S3..."
rclone sync ${S3_REMOTE}/ALPHAQUBIT/generate_data.py . -v
rclone sync ${S3_REMOTE}/ALPHAQUBIT/make_all_pretraining_noise.py . -v
rclone sync ${S3_REMOTE}/ALPHAQUBIT/run_create_all_samples.py . -v
rclone sync ${S3_REMOTE}/ALPHAQUBIT/simulator ./simulator -v
rclone sync ${S3_REMOTE}/ALPHAQUBIT/google_qec_simulator ./google_qec_simulator -v
rclone sync ${S3_REMOTE}/ALPHAQUBIT/ai_models ./ai_models -v
rclone sync ${S3_REMOTE}/ALPHAQUBIT/configs ./configs -v

echo "Running pipeline..."
python3 make_all_pretraining_noise.py \
    --experiment-root ${GOOGLE_DATA} \
    --use-dem-stim \
    --exp-samples 50000 \
    --soft-shots 100000 \
    --soft-device auto

echo "Done! Results in: ${WORK_DIR}/pretrain_data/"
