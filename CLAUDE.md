# Consistency Model for CIFAR-10

## Project Overview

Implementation of Consistency Training (CT) from Song et al. (2023) for unconditional one-step image generation on CIFAR-10 (32×32). The model learns to map any point on a diffusion ODE trajectory back to clean data, enabling single-step generation.

## File Structure

- `model.py` — UNet with time conditioning + ConsistencyModel wrapper (c_skip/c_out parameterization)
- `train.py` — Consistency Training loop with DDP multi-GPU support, curriculum scheduling, EMA target, pseudo-Huber loss
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
| ρ (Karras schedule) | 7.0 |
| Effective batch size | 4096 |
| Micro batch size | 512 |
| Gradient accumulation | 2 steps per GPU (with 4 GPUs) |
| GPUs | 4 (scales automatically) |
| Learning rate | 1e-4 (Adam) |
| EMA decay | 0.999 |
| Total steps | 800k |
| N curriculum | 2 → 150 |
| Pseudo-Huber c | 0.00054 * √(3·32·32) |

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train (4 GPUs)
torchrun --nproc_per_node=4 train.py

# Train (single GPU — grad_accum auto-adjusts to 8)
torchrun --nproc_per_node=1 train.py

# Sample (after training)
python sample.py --checkpoint checkpoints/consistency_final.pt --n_samples 64 --output samples.png
```

## Development Notes

- **DDP**: Multi-GPU training via `DistributedDataParallel`; `torchrun` sets up process groups automatically
- **Gradient accumulation**: Effective batch 2048 = 256 micro-batch × 2 accum × 4 GPUs (auto-adjusts with GPU count)
- **`no_sync` optimization**: Intermediate micro-batches skip all-reduce; only the final micro-batch synchronizes gradients
- **AMP**: Mixed precision training with `torch.amp.autocast` and `GradScaler`
- **Target model**: Not wrapped in DDP (no gradients, EMA-only); stays as a plain module
- Checkpoints save to `./checkpoints/` every 50k steps (includes online, target, optimizer, and scaler state dicts)
- Checkpoints store unwrapped `online.module.state_dict()` so they load directly in `sample.py` without key mismatches
- CIFAR-10 downloads to `./data/` on first run (rank 0 downloads, others wait at barrier)
- Training logs loss every 500 steps (rank 0 only)
- Use target (EMA) model weights for sampling
