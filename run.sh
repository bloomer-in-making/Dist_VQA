#!/bin/bash

set -e

TASK=${1:-xyz}
PHASE=${2:-all}
SEED=${3:-2025} 

cd "$(dirname "$0")"

echo "=========================================="
echo "VQA task run"
echo "Task: $TASK"
echo "Phase: $PHASE"
echo "Seed: $SEED"
echo "=========================================="

# seed override command
SEED_CMD="--set phases.phase1.config.seed=$SEED --set phases.phase2.config.seed=$SEED"

if [ "$TASK" = "xyz" ]; then
    if [ "$PHASE" = "all" ]; then
        echo "Phase 1 running..."
        python main.py --task xyz --config configs/xyz.yaml --phase phase1 $SEED_CMD

        echo ""
        echo "Phase 2 running..."

        BEST_PATH=$(SEED_ENV="$SEED" python - << 'EOF'
import os
import yaml

cfg = yaml.safe_load(open("configs/xyz.yaml"))

phase_cfg = cfg.get("phases", {}).get("phase1", {}).get("config", {})

seed = int(os.environ.get("SEED_ENV", phase_cfg.get("seed", cfg.get("seed", 2025))))
qubit = phase_cfg.get("qubit", cfg.get("qubit", 20))

model_dir = os.path.join(
    "results", "xyz_continuous_grad",
    f"qubit{qubit}",
    "MLP",
    f"seed{seed}",
)

print(os.path.join(model_dir, "best.pt"))
EOF
)

        if [ ! -f "$BEST_PATH" ]; then
            echo "Warning: could not find the auto-computed best.pt path:"
            echo "  $BEST_PATH"
            exit 1
        fi

        python main.py --task xyz --config configs/xyz.yaml --phase phase2 \
            --set phases.phase2.config.checkpoint_path="$BEST_PATH"
    elif [ "$PHASE" = "phase1" ]; then
        python main.py --task xyz --config configs/xyz.yaml --phase phase1 $SEED_CMD
    elif [ "$PHASE" = "phase2" ]; then
        BEST_PATH=${4:-$(find . -name "best.pt" -path "*/qubit*/*seed$SEED*" | head -1)}
        
        if [ -z "$BEST_PATH" ]; then
            echo "Error: checkpoint_path is required."
            echo "Usage: ./run.sh xyz phase2 <seed> <checkpoint_path>"
            exit 1
        fi
        python main.py --task xyz --config configs/xyz.yaml --phase phase2 --set phases.phase2.config.checkpoint_path="$BEST_PATH" $SEED_CMD
    fi
elif [ "$TASK" = "qaoa" ]; then
    if [ "$PHASE" = "all" ]; then
        echo "Phase 1 running..."
        python main.py --task qaoa --config configs/qaoa.yaml --phase phase1 $SEED_CMD

        echo ""
        echo "Phase 2 running..."

        BEST_PATH=$(SEED_ENV="$SEED" python - << 'EOF'
import os, yaml

cfg = yaml.safe_load(open("configs/qaoa.yaml"))

phase_cfg = cfg.get("phases", {}).get("phase1", {}).get("config", {})

seed = int(os.environ.get("SEED_ENV", phase_cfg.get("seed", cfg.get("seed", 2025))))
n = phase_cfg.get("n", cfg.get("n", 10))
p = phase_cfg.get("p", cfg.get("p", 4))

model_dir = os.path.join(
    "results", "qaoa_grad",
    f"train_hypernet_n{n}_p{p}_MLP",
    f"seed{seed}",
)

print(os.path.join(model_dir, "best.pt"))
EOF
)

        if [ ! -f "$BEST_PATH" ]; then
            echo "Warning: the auto-generated best.pt path does not exist:"
            echo " → $BEST_PATH"
            exit 1
        fi

        python main.py --task qaoa --config configs/qaoa.yaml --phase phase2 \
            --set phases.phase2.config.checkpoint_path="$BEST_PATH" $SEED_CMD
    fi
else
    echo "Error: Unknown task '$TASK'. Use 'xyz' or 'qaoa'"
    exit 1
fi

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="

