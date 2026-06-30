#!/bin/bash

ENVS=("Acrobot-v1" "CartPole-v1" "LunarLander-v3" "MountainCar-v0")
MODELS=("dqn" "fqf" "iqn" "mcfqf" "mcqrdqn" "qrdqn")
SEED=42

for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Starting: env=$env, model=$model"
        uv run train.py --env-id "$env" --model-id "$model" --seed $SEED &
    done
done
