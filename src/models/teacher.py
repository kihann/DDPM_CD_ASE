import utils
import torch
from diffusers import DDPMPipeline
import math

DEVICE = utils.DEVICE
MODEL_ID = utils.MODEL_ID
# TIMESTEP from utils is the DDPM's total training timesteps (e.g., 1000)
# utils.TIMESTEP 

# Constants for Consistency Distillation (CD) based on "Consistency Models" paper
SIGMA_MIN = 0.002
SIGMA_MAX = 1.5
RHO = 7.0
N_CD_STEPS = 18  # Number of discretization steps for CD (N in the paper)

pipeline = DDPMPipeline.from_pretrained(MODEL_ID, use_safetensors=False).to(DEVICE)
unet = pipeline.unet.eval()  # Teacher U-Net
scheduler = pipeline.scheduler # This is a DDPMScheduler instance
# Ensure scheduler is configured for the total number of DDPM timesteps
# Default DDPMPipeline usually has scheduler.timesteps set during __init__ or by .from_pretrained
# If not, ensure scheduler.set_timesteps(utils.TIMESTEP) was effectively called.
# We need scheduler.alphas_cumprod which corresponds to utils.TIMESTEP steps.

# Precompute sigmas for CD schedule (t_i in the paper)
# These are N_CD_STEPS points, indexed 0 to N_CD_STEPS-1
_sigmas_for_cd = torch.tensor([
    (SIGMA_MIN**(1.0/RHO) + i/(N_CD_STEPS-1) * (SIGMA_MAX**(1.0/RHO) - SIGMA_MIN**(1.0/RHO)))**RHO
    for i in range(N_CD_STEPS)
]).float().to(DEVICE)

def map_sigma_to_scheduler_timestep(sigma_val_scalar: float, scheduler_instance, device_to_use) -> int: # device_to_use 인자 추가
    """Maps a continuous sigma value to the closest discrete DDPMScheduler timestep index."""
    if not hasattr(scheduler_instance, 'alphas_cumprod'):
        # ... (alphas_cumprod_tensor 로드 로직, device_to_use 사용) ...
        if hasattr(scheduler_instance, 'config') and 'alphas_cumprod' in scheduler_instance.config:
             alphas_cumprod_source = scheduler_instance.config.alphas_cumprod
        elif hasattr(scheduler_instance, '_alphas_cumprod'): 
             alphas_cumprod_source = scheduler_instance._alphas_cumprod
        else:
            raise ValueError("Scheduler must have 'alphas_cumprod' attribute or accessible config.")
        
        if not isinstance(alphas_cumprod_source, torch.Tensor):
            alphas_cumprod_tensor = torch.tensor(alphas_cumprod_source, device=device_to_use, dtype=torch.float32)
        else:
            alphas_cumprod_tensor = alphas_cumprod_source.to(device_to_use)
    else:
        alphas_cumprod_tensor = scheduler_instance.alphas_cumprod.to(device_to_use)


    target_alpha_cumprod_scalar = 1.0 - sigma_val_scalar**2
    
    min_alpha_cumprod = alphas_cumprod_tensor.min()
    max_alpha_cumprod = alphas_cumprod_tensor.max()
    
    target_alpha_cumprod_tensor_device = torch.tensor(target_alpha_cumprod_scalar, device=device_to_use, dtype=torch.float32)

    clamped_target_alpha_cumprod = torch.clamp(target_alpha_cumprod_tensor_device, 
                                               min_alpha_cumprod, 
                                               max_alpha_cumprod)

    abs_diff = (alphas_cumprod_tensor - clamped_target_alpha_cumprod).abs()
    closest_timestep_idx = abs_diff.argmin().item()
    return closest_timestep_idx

# generate_consistency_distillation_pair_rigorous 함수 내에서 호출 시 DEVICE 전달
# k_n_plus_1 = map_sigma_to_scheduler_timestep(sigma_cd_n_plus_1_val.item(), ddpm_scheduler, DEVICE)

@torch.no_grad()
def generate_consistency_distillation_pair_rigorous(
    x0: torch.Tensor,
    teacher_unet_model, 
    ddpm_scheduler
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    global DEVICE 
    B = x0.shape[0]
    current_device = x0.device

    rand_n_idx_scalar = torch.randint(0, N_CD_STEPS - 1, (1,)).item()
    sigma_cd_n_val = _sigmas_for_cd[rand_n_idx_scalar]
    sigma_cd_n_plus_1_val = _sigmas_for_cd[rand_n_idx_scalar + 1]

    sigma_cd_n = torch.full((B,), sigma_cd_n_val.item(), device=current_device, dtype=torch.float32)
    sigma_cd_n_plus_1 = torch.full((B,), sigma_cd_n_plus_1_val.item(), device=current_device, dtype=torch.float32)

    k_n_plus_1 = map_sigma_to_scheduler_timestep(sigma_cd_n_plus_1_val.item(), ddpm_scheduler, current_device)
    k_n_plus_1_tensor = torch.full((B,), k_n_plus_1, device=current_device, dtype=torch.long)

    noise_for_ddpm = torch.randn_like(x0, device=current_device)
    xt_k_n_plus_1 = ddpm_scheduler.add_noise(x0, noise_for_ddpm, k_n_plus_1_tensor)

    epsilon_teacher = teacher_unet_model(xt_k_n_plus_1, k_n_plus_1_tensor).sample
    
    if not hasattr(ddpm_scheduler, 'alphas_cumprod'):
        # ... (이전과 동일한 alphas_cumprod 로드 로직) ...
        if hasattr(ddpm_scheduler, 'config') and 'alphas_cumprod' in ddpm_scheduler.config:
             alphas_cumprod_source = ddpm_scheduler.config.alphas_cumprod
        elif hasattr(ddpm_scheduler, '_alphas_cumprod'): 
             alphas_cumprod_source = ddpm_scheduler._alphas_cumprod
        else:
            raise ValueError("Scheduler must have 'alphas_cumprod' attribute or accessible config.")
        if not isinstance(alphas_cumprod_source, torch.Tensor):
            alphas_cumprod_tensor_sched = torch.tensor(alphas_cumprod_source, device=current_device, dtype=torch.float32)
        else:
            alphas_cumprod_tensor_sched = alphas_cumprod_source.to(current_device)
    else:
        alphas_cumprod_tensor_sched = ddpm_scheduler.alphas_cumprod.to(current_device)

    sqrt_one_minus_alpha_bar_k_n_plus_1 = (1.0 - alphas_cumprod_tensor_sched[k_n_plus_1]).sqrt()
    # sigma_eff_at_k_n_plus_1 정의
    sigma_eff_at_k_n_plus_1 = torch.max(sqrt_one_minus_alpha_bar_k_n_plus_1, torch.tensor(1e-9, device=current_device))

    sigma_cd_n_b = sigma_cd_n.view(B, 1, 1, 1)
    sigma_cd_n_plus_1_b = sigma_cd_n_plus_1.view(B, 1, 1, 1)
    
    # ode_term_coefficient 정의
    ode_term_coefficient = (sigma_cd_n_b - sigma_cd_n_plus_1_b) * \
                    (-sigma_cd_n_plus_1_b) * \
                    (-epsilon_teacher / sigma_eff_at_k_n_plus_1) 
    
    # xt_sigma_cd_n_hat_phi 정의
    xt_sigma_cd_n_hat_phi = xt_k_n_plus_1 + ode_term_coefficient
    
    # --- 로깅 코드 시작 ---
    # 모든 필요한 변수가 정의된 이후에 로깅을 수행합니다.
    # 확률적 로깅 (예: 0.1% 확률로 로깅, 또는 특정 스텝마다 로깅)
    # train.py에서 step 수를 이 함수에 전달하거나, 여기서 자체적으로 카운트/랜덤 샘플링 가능
    # 여기서는 간단히 확률적으로 로깅
    #if torch.rand(1).item() < 0.1: # 1% 확률로 로깅 (또는 필요에 따라 조절)
    #   print(f"Debug CD Pair Gen (Rigorous):")
    #   print(f"  sigma_cd_n: {sigma_cd_n_val.item():.4f}, sigma_cd_n+1: {sigma_cd_n_plus_1_val.item():.4f}")
    #   print(f"  k_n+1 (DDPM ts): {k_n_plus_1}")
    #   print(f"  sqrt_1-alpha_bar (sigma_eff): {sigma_eff_at_k_n_plus_1.item():.4e}") # 이제 안전
    #   print(f"  epsilon_teacher norm: {torch.norm(epsilon_teacher).item():.4f}")
       # ode_term_coefficient는 텐서일 수 있으므로 norm 또는 특정 요소 출력
    #   print(f"  ode_term_coeff norm: {torch.norm(ode_term_coefficient).item():.4e}") 
    #   print(f"  xt_k_n_plus_1 norm: {torch.norm(xt_k_n_plus_1).item():.2f}, xt_sigma_cd_n_hat_phi norm: {torch.norm(xt_sigma_cd_n_hat_phi).item():.2f}")
    #   if torch.isnan(xt_sigma_cd_n_hat_phi).any() or torch.isinf(xt_sigma_cd_n_hat_phi).any():
    #       print("!!!! NaN or Inf detected in xt_sigma_cd_n_hat_phi !!!!")
    #   elif torch.norm(xt_sigma_cd_n_hat_phi) > 1e5: # 임계값은 조절 가능
    #       print(f"!!!! Large norm detected in xt_sigma_cd_n_hat_phi: {torch.norm(xt_sigma_cd_n_hat_phi).item()} !!!!")
    # --- 로깅 코드 끝 ---

    return xt_k_n_plus_1, sigma_cd_n_plus_1, xt_sigma_cd_n_hat_phi, sigma_cd_n