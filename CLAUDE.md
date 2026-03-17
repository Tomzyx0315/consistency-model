# Consistency Model for CIFAR-10

## Project Overview

Implementation of Consistency Training (CT) from Song et al. (2023) for unconditional one-step image generation on CIFAR-10 (32×32). The model learns to map any point on a diffusion ODE trajectory back to clean data, enabling single-step generation.

## File Structure

- `model.py` — UNet with time conditioning + ConsistencyModel wrapper (c_skip/c_out parameterization)
- `train.py` — Consistency Training loop with curriculum scheduling, EMA target, pseudo-Huber loss
- `sample.py` — One-step sampling: z * σ_max → f_θ → image grid
- `requirements.txt` — torch, torchvision, tqdm

## Architecture

- **UNet**: Channels [128, 256, 256], 2 ResBlocks per level, self-attention at 16×16 resolution
- **Time conditioning**: Sinusoidal encoding → 2-layer MLP, injected into ResBlocks
- **Consistency parameterization**: `f_θ(x, t) = c_skip(t) * x + c_out(t) * F_θ(x, t)`

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| σ_data | 0.5 |
| σ_min / σ_max | 0.002 / 80.0 |
| Batch size | 512 |
| Learning rate | 1e-4 (Adam) |
| EMA decay | 0.999 |
| Total steps | 800k |
| N curriculum | 2 → 150 |
| Pseudo-Huber c | 0.00054 * √(3·32·32) |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train
python train.py

# Sample (after training)
python sample.py --checkpoint checkpoints/consistency_final.pt --n_samples 64 --output samples.png
```

## Development Notes

- Checkpoints save to `./checkpoints/` every 50k steps
- CIFAR-10 downloads to `./data/` on first run
- Training logs loss every 500 steps
- Use target (EMA) model weights for sampling
