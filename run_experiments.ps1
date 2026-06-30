$ENVS = @("Acrobot-v1", "CartPole-v1", "LunarLander-v3", "MountainCar-v0")
$MODELS = @("dqn", "fqf", "iqn", "mcfqf", "mcqrdqn", "qrdqn")
$SEED = 42

foreach ($env in $ENVS) {
    foreach ($model in $MODELS) {
        Write-Host "Starting: env=$env, model=$model"
        Start-Process -FilePath "uv" -ArgumentList "run", "train.py", "--env-id", $env, "--model-id", $model, "--seed", $SEED
    }
}
