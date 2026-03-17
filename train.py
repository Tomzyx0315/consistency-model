"""Consistency Training for CIFAR-10."""

import copy
import math
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from model import ConsistencyModel


# ──────────────────────────────────────────────
# Hyperparameters
# ──────────────────────────────────────────────
SIGMA_DATA = 0.5
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0
RHO = 7.0
S0 = 2          # initial discretization steps
S1 = 150        # final discretization steps
EMA_DECAY = 0.999
BATCH_SIZE = 512
MICRO_BATCH = 64
GRAD_ACCUM_STEPS = BATCH_SIZE // MICRO_BATCH  # 8
LR = 1e-4
TOTAL_STEPS = 800_000
LOG_EVERY = 500
SAVE_EVERY = 50_000
DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"
D = 3 * 32 * 32  # data dimensionality
HUBER_C = 0.00054 * math.sqrt(D)


def karras_schedule(n_steps, sigma_min=SIGMA_MIN, sigma_max=SIGMA_MAX, rho=RHO):
    """Return n_steps+1 sigma values from the Karras noise schedule (descending)."""
    ramp = torch.linspace(0, 1, n_steps + 1)
    min_inv = sigma_min ** (1.0 / rho)
    max_inv = sigma_max ** (1.0 / rho)
    sigmas = (max_inv + ramp * (min_inv - max_inv)) ** rho
    return sigmas


def n_steps_schedule(step, total_steps, s0=S0, s1=S1):
    """Curriculum for N: starts at s0, increases to s1 over training."""
    k = step
    K = total_steps
    N = math.ceil(math.sqrt(k / K * ((s1 + 1) ** 2 - s0 ** 2) + s0 ** 2) - 1) + 1
    return max(N, s0)


def pseudo_huber_loss(x, y, c=HUBER_C):
    """Pseudo-Huber loss: sqrt((x-y)^2 + c^2) - c, averaged over batch."""
    diff = (x - y).reshape(x.shape[0], -1)
    loss = (diff ** 2 + c ** 2).sqrt() - c
    return loss.mean()


def get_dataloader():
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1, 1]
    ])
    dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return loader


def infinite_loader(loader):
    """Yield batches forever."""
    while True:
        yield from loader


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(CKPT_DIR, exist_ok=True)

    # Models
    online = ConsistencyModel(sigma_data=SIGMA_DATA, sigma_min=SIGMA_MIN).to(device)
    target = copy.deepcopy(online)
    target.requires_grad_(False)

    param_count = sum(p.numel() for p in online.parameters())
    print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')
    loader = get_dataloader()
    data_iter = infinite_loader(loader)

    running_loss = 0.0

    for step in tqdm(range(1, TOTAL_STEPS + 1), desc="Training"):
        # Current number of discretization steps
        N = n_steps_schedule(step, TOTAL_STEPS)
        sigmas = karras_schedule(N).to(device)

        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM_STEPS):
            x, _ = next(data_iter)
            x = x.to(device)

            # Sample random index n in {0, ..., N-2} -> adjacent pair (t_{n+1}, t_n)
            n = torch.randint(0, N - 1, (x.shape[0],), device=device)
            t_next = sigmas[n]      # t_{n+1} (larger sigma)
            t_curr = sigmas[n + 1]  # t_n     (smaller sigma)

            # Same noise, different scales
            z = torch.randn_like(x)
            x_next = x + t_next[:, None, None, None] * z   # noisier
            x_curr = x + t_curr[:, None, None, None] * z   # less noisy

            with torch.amp.autocast('cuda'):
                # Online model: f_θ(x_{n+1}, t_{n+1})
                pred_online = online(x_next, t_next)

                # Target model (no grad): f_{θ⁻}(x_n, t_n)
                with torch.no_grad():
                    pred_target = target(x_curr, t_curr)

                loss = pseudo_huber_loss(pred_online, pred_target)
                loss = loss / GRAD_ACCUM_STEPS

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        # EMA update of target model
        with torch.no_grad():
            for p_online, p_target in zip(online.parameters(), target.parameters()):
                p_target.data.lerp_(p_online.data, 1.0 - EMA_DECAY)

        running_loss += accum_loss

        if step % LOG_EVERY == 0:
            avg = running_loss / LOG_EVERY
            print(f"Step {step}/{TOTAL_STEPS} | N={N} | Loss: {avg:.4f}")
            running_loss = 0.0

        if step % SAVE_EVERY == 0:
            path = os.path.join(CKPT_DIR, f"consistency_step{step}.pt")
            torch.save({
                "step": step,
                "online_state_dict": online.state_dict(),
                "target_state_dict": target.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
            }, path)
            print(f"Saved checkpoint: {path}")

    # Save final
    path = os.path.join(CKPT_DIR, "consistency_final.pt")
    torch.save({
        "step": TOTAL_STEPS,
        "online_state_dict": online.state_dict(),
        "target_state_dict": target.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
    }, path)
    print(f"Training complete. Final checkpoint: {path}")


if __name__ == "__main__":
    train()
