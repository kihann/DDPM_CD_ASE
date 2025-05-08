import utils
import torch
from diffusers import DDPMPipeline

DEVICE = utils.DEVICE
MODEL_ID = utils.MODEL_ID
TIMESTEP = utils.TIMESTEP

pipeline = DDPMPipeline.from_pretrained(MODEL_ID, use_safetensors=False).to(DEVICE)

unet = pipeline.unet.eval()
scheduler = pipeline.scheduler
scheduler.set_timesteps(TIMESTEP)

def sample_different_indices(length, size):
    idx1 = torch.randint(0, length, (size,))
    idx2 = torch.randint(0, length, (size,))
    while (idx1 == idx2).any():
        mask = idx1 == idx2
        idx2[mask] = torch.randint(0, length, (mask.sum(),))
    return idx1, idx2

@torch.no_grad()
def generate_consistency_pair(x0: torch.Tensor) -> tuple[torch.Tensor, torch.LongTensor, torch.Tensor, torch.LongTensor]:
    """
    Generate noisy image pairs at two different timesteps using shared noise.
    
    Args:
        x0 (torch.Tensor): Clean images, shape (B, C, H, W)
    Returns:
        xt1, t1, xt2, t2
    """
    B = x0.size(0)
    noise = torch.randn_like(x0)  # Shared noise

    t1_idx, t2_idx = sample_different_indices(len(scheduler.timesteps), B)

    t1 = scheduler.timesteps[t1_idx].to(DEVICE)
    t2 = scheduler.timesteps[t2_idx].to(DEVICE)

    xt1 = scheduler.add_noise(x0, noise, t1)
    xt2 = scheduler.add_noise(x0, noise, t2)

    return xt1, t1, xt2, t2