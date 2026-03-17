"""Consistency Training for CIFAR-10 (Multi-GPU DDP)."""

import copy
import math
import os
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
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
BATCH_SIZE = 4096
MICRO_BATCH = 512
LR = 1e-4
TOTAL_STEPS = 100_000
LOG_EVERY = 500
SAVE_EVERY = 20_000
DATA_DIR = "./data"
CKPT_DIR = "./checkpoints"
D = 3 * 32 * 32  # data dimensionality
HUBER_C = 0.00054 * math.sqrt(D)


def setup_ddp():
    """Initialize DDP process group and set the CUDA device."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    """Destroy the DDP process group."""
    dist.destroy_process_group()


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
    sampler = DistributedSampler(dataset)
    loader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )
    return loader, sampler


def infinite_loader(loader, sampler):
    """Yield batches forever, re-shuffling each epoch via the sampler."""
    epoch = 0
    while True:
        sampler.set_epoch(epoch)
        yield from loader
        epoch += 1


def train():
    # DDP setup
    local_rank = setup_ddp()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    grad_accum_steps = BATCH_SIZE // (MICRO_BATCH * world_size)

    if rank == 0:
        print(f"DDP training with {world_size} GPU(s)")
        print(f"Effective batch size: {BATCH_SIZE} = {MICRO_BATCH} micro-batch × {grad_accum_steps} accum × {world_size} GPUs")
        os.makedirs(CKPT_DIR, exist_ok=True)

    # Models
    online = ConsistencyModel(sigma_data=SIGMA_DATA, sigma_min=SIGMA_MIN).to(device)
    target = copy.deepcopy(online)
    target.requires_grad_(False)

    # Wrap online model with DDP
    online = DDP(online, device_ids=[local_rank])

    if rank == 0:
        param_count = sum(p.numel() for p in online.parameters())
        print(f"Model parameters: {param_count:,}")

    optimizer = torch.optim.Adam(online.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda')

    # Rank 0 downloads data first, then others proceed
    if rank == 0:
        datasets.CIFAR10(root=DATA_DIR, train=True, download=True)
    dist.barrier()

    loader, sampler = get_dataloader()
    data_iter = infinite_loader(loader, sampler)

    running_loss = 0.0

    pbar = tqdm(range(1, TOTAL_STEPS + 1), desc="Training", disable=(rank != 0))
    for step in pbar:
        # Current number of discretization steps
        N = n_steps_schedule(step, TOTAL_STEPS)
        sigmas = karras_schedule(N).to(device)

        optimizer.zero_grad()
        accum_loss = 0.0

        for micro_step in range(grad_accum_steps):
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

            # Skip all-reduce on intermediate micro-batches
            sync_context = online.no_sync() if micro_step < grad_accum_steps - 1 else nullcontext()

            with sync_context:
                with torch.amp.autocast(device_type='cuda'):
                    # Online model: f_θ(x_{n+1}, t_{n+1})
                    pred_online = online(x_next, t_next)

                    # Target model (no grad): f_{θ⁻}(x_n, t_n)
                    with torch.no_grad():
                        pred_target = target(x_curr, t_curr)

                    loss = pseudo_huber_loss(pred_online, pred_target)
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.step(optimizer)
        scaler.update()

        # EMA update of target model
        with torch.no_grad():
            for p_online, p_target in zip(online.module.parameters(), target.parameters()):
                p_target.data.lerp_(p_online.data, 1.0 - EMA_DECAY)

        running_loss += accum_loss

        if step % LOG_EVERY == 0 and rank == 0:
            avg = running_loss / LOG_EVERY
            print(f"Step {step}/{TOTAL_STEPS} | N={N} | Loss: {avg:.4f}")
            running_loss = 0.0

        if step % SAVE_EVERY == 0:
            if rank == 0:
                path = os.path.join(CKPT_DIR, f"consistency_step{step}.pt")
                torch.save({
                    "step": step,
                    "online_state_dict": online.module.state_dict(),
                    "target_state_dict": target.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }, path)
                print(f"Saved checkpoint: {path}")
            dist.barrier()

    # Save final
    if rank == 0:
        path = os.path.join(CKPT_DIR, "consistency_final.pt")
        torch.save({
            "step": TOTAL_STEPS,
            "online_state_dict": online.module.state_dict(),
            "target_state_dict": target.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }, path)
        print(f"Training complete. Final checkpoint: {path}")

    cleanup_ddp()


if __name__ == "__main__":
    train()
