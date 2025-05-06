import utils
from diffusers import DDPMPipeline

DEVICE = utils.DEVICE
MODEL_ID = utils.MODEL_ID

# load model and scheduler
pipe = DDPMPipeline.from_pretrained(MODEL_ID, use_safetensors=False).to(DEVICE)

# run pipeline in inference (sample random noise and denoise)
image = pipe().images[0]

# save image
image.save("ddpm_generated_image.png")