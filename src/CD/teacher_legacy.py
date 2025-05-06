import utils
import torch

from diffusers import DDPMPipeline

DEVICE = utils.DEVICE
MODEL_ID = utils.MODEL_ID

pipeline = DDPMPipeline.from_pretrained(MODEL_ID)
pipeline.to(DEVICE)

unet = pipeline.unet.eval()
scheduler = pipeline.scheduler

x0 = torch.randn(1, 3, 256, 256).to(DEVICE)
noise = torch.randn_like(x0)
timestep = torch.tensor([500], dtype=torch.long, device=DEVICE)

xt = scheduler.add_noise(original_samples=x0, noise=noise, timesteps=timestep)

def fn_consistency_distillation(teacher_pred):
    alpha_bar_t = scheduler.alphas_cumprod[timestep].view(-1, 1, 1, 1).to(xt.device)
    
    return (xt - (1-alpha_bar_t).sqrt() * teacher_pred) / alpha_bar_t.sqrt()

with torch.no_grad():
    teacher_pred = unet(xt, timestep).sample
    x0_teacher = fn_consistency_distillation(teacher_pred)

print(f"xt shape: {xt.shape}")
print(f"teacher_pred shape: {teacher_pred.shape}")
print(f"x0_teacher shape: {x0_teacher.shape}")