"""One-step sampling from a trained consistency model."""

import argparse

import torch
from torchvision.utils import save_image

from model import ConsistencyModel


SIGMA_DATA = 0.5
SIGMA_MIN = 0.002
SIGMA_MAX = 80.0


def sample(checkpoint_path, n_samples=64, output_path="samples.png", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = ConsistencyModel(sigma_data=SIGMA_DATA, sigma_min=SIGMA_MIN).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    # Use target (EMA) model for sampling
    state_dict = ckpt.get("target_state_dict", ckpt.get("online_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    print(f"Generating {n_samples} samples...")

    with torch.no_grad():
        # z ~ N(0, I) scaled by sigma_max
        z = torch.randn(n_samples, 3, 32, 32, device=device) * SIGMA_MAX
        t = torch.full((n_samples,), SIGMA_MAX, device=device)

        # One-step generation
        x = model(z, t)

        # Clamp to [-1, 1] and rescale to [0, 1] for saving
        x = x.clamp(-1, 1) * 0.5 + 0.5

    nrow = int(n_samples ** 0.5)
    save_image(x, output_path, nrow=nrow)
    print(f"Saved sample grid to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample from consistency model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="samples.png", help="Output image path")
    args = parser.parse_args()

    sample(args.checkpoint, args.n_samples, args.output)
