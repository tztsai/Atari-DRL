# Atari DRL

Use PPO (Proximal Policy Optimization) to play Atari games!

## How to Play

Train the RL agent:

```bash
python main.py --env-name "Breakout-v3" --num-steps 2048 --num-processes 1 --lr 3e-4 --entropy-coef 0 --value-loss-coef 0.5 --ppo-epoch 10 --num-mini-batch 32 --gamma 0.99 --gae-lambda 0.95 --num-env-steps 1000000 --use-linear-lr-decay --use-proper-time-limits
```

Let the agent play!

```bash
python enjoy.py --load-dir trained_models --env-name "Breakout-v3"
```
