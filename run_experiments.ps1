$ENVS = @("Acrobot-v1", "CartPole-v1", "MountainCar-v0", "LunarLander-v3")
$MODELS = @("dqn", "qrdqn", "mcqrdqn")
$SEED = 42

foreach ($env in $ENVS) {
    foreach ($model in $MODELS) {
        Write-Host "Starting: env=$env, model=$model"
        Start-Process -FilePath "uv" -ArgumentList "run", "train.py", "--env-id", $env, "--model-id", $model, "--seed", $SEED
    }
}
