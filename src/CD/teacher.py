import utils
import torch
from diffusers import DDPMPipeline

DEVICE = utils.DEVICE
MODEL_ID = utils.MODEL_ID
TIMESTEP = 1000

pipeline = DDPMPipeline.from_pretrained(MODEL_ID, use_safetensors=False).to(DEVICE)

unet = pipeline.unet.eval()
scheduler = pipeline.scheduler
scheduler.set_timesteps(TIMESTEP)

def generate_consistency_pair(x0: torch.Tensor):
    """
    Args:
        x0 (torch.Tensor): Original clean image batch, shape (B, C, H, W)
    Returns:
        xt1, t1, xt2, t2: Noisy image pairs and their timesteps
    """
    noise = torch.randn_like(x0)

    B = x0.size(0)
    t1_idx = torch.randint(0, len(scheduler.timesteps), (B,))
    t2_idx = torch.randint(0, len(scheduler.timesteps), (B,))

    while (t1_idx == t2_idx).any():
        mask = t1_idx == t2_idx
        t2_idx[mask] = torch.randint(0, len(scheduler.timesteps), (mask.sum(),))

    t1 = scheduler.timesteps[t1_idx].to(DEVICE)
    t2 = scheduler.timesteps[t2_idx].to(DEVICE)

    xt1 = scheduler.add_noise(x0, noise, t1)
    xt2 = scheduler.add_noise(x0, noise, t2)

    return xt1, t1, xt2, t2