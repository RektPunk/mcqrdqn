#!/bin/bash

ENVS=("Acrobot-v1" "CartPole-v1" "MountainCar-v0" "LunarLander-v3")
MODELS=("dqn" "qrdqn" "mcqrdqn")
SEED=42

for env in "${ENVS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "Starting: env=$env, model=$model"
        uv run train.py --env-id "$env" --model-id "$model" --seed $SEED &
    done
done
