import torch
import torchvision
import matplotlib.pyplot as plt
from CD import student # 학생 모델 아키텍처
from CD import teacher # 스케줄러 등을 가져오기 위해 (옵션)
import utils # DEVICE, TIMESTEP 등

DEVICE = utils.DEVICE
TIMESTEP = utils.TIMESTEP # utils.py에 정의된 값

def sample_images(model_path, num_images=4, num_steps=50, guidance_scale=None, skip_config=None): # skip_config는 ASE 관련 설정
    # 1. 모델 초기화 및 가중치 로드
    # diffusion_steps는 학습 시 사용한 값과 동일하게, 
    # skip_exponent, skip_thresholds 등도 학습 시 사용했거나 실험하려는 값으로 설정
    # UNetStudent 생성 시 train.py와 동일한 파라미터 사용 (base_channels 등)
    unet_student = student.UNetStudent(
        diffusion_steps=TIMESTEP, 
        skip_exponent=2.5, # 학습 시 사용한 값
        # skip_thresholds=... # 필요시 명시적으로 설정
    ).to(DEVICE)
    unet_student.load_state_dict(torch.load(model_path, map_location=DEVICE))
    unet_student.eval()

    # 2. 스케줄러 준비 (DDPMScheduler 또는 Consistency Model에 맞는 스케줄러)
    # 여기서는 DDPM 스케줄러를 예시로 사용 (Consistency Models 논문 참고하여 수정 가능)
    # teacher.scheduler를 재활용하거나, 새로 DDPMScheduler를 설정할 수 있습니다.
    # 만약 Consistency Models 논문의 1-step/few-step 샘플링을 직접 구현한다면,
    # 스케줄러의 역할이 다를 수 있습니다.
    
    # Consistency Model 1-step 샘플링 (가장 간단한 형태)
    # CM 논문에서는 x_T ~ N(0, T^2 I) 에서 시작하여 f_theta(x_T, T)로 x_0를 얻습니다.
    # 여기서 T는 가장 큰 타임스텝 인덱스 또는 sigma_max에 해당합니다.
    
    images = []
    with torch.no_grad():
        for _ in range(num_images):
            # 초기 노이즈 생성 (예: CelebA 256x256, 3채널)
            # Consistency Models 논문에서는 T를 큰 값(예: 80)으로 설정하고 x_T ~ N(0, T^2 I)를 사용합니다.
            # DDPM의 마지막 타임스텝의 노이즈를 사용할 수도 있습니다.
            # 아래는 DDPM 방식의 초기 노이즈입니다.
            noise_shape = (1, 3, 256, 256) # (batch, channels, height, width)
            xt = torch.randn(noise_shape, device=DEVICE)
            
            # 가장 큰 타임스텝 값 (예: 학습 시 TIMESTEP - 1)
            # 또는 CM 논문에서 사용하는 t=T (sigma_max)
            # t_tensor = torch.tensor([TIMESTEP - 1], device=DEVICE, dtype=torch.long) 
            # 혹은 CM에서 사용하는 t_max 값
            t_max = torch.tensor([utils.TIMESTEP-1], device=DEVICE).long() # 예시, 실제로는 CM 논문상의 T값을 사용

            # 1-step 생성: 학생 모델 사용
            # return_all=False로 설정하여 최종 (또는 특정 exit) 출력을 받습니다.
            # Adaptive Score Estimation이 적용된 샘플링을 위해서는,
            # model의 forward가 t 값에 따라 내부적으로 블록을 스킵합니다.
            pred_x0 = unet_student(xt, t_max, return_all=False) 
            
            # 만약 다중 exit 중 하나를 선택적으로 사용하려면, forward 호출 후 로직 추가
            # 예: if t_max > 특정값: pred_x0 = unet_student(xt, t_max, return_all=True)[-2] # 마지막에서 두번째 exit 사용
            
            # 이미지 후처리 ([-1, 1] 범위를 [0, 1]로 변환 등)
            pred_x0 = (pred_x0.clamp(-1, 1) + 1) / 2
            images.append(pred_x0.cpu())

    # 이미지 저장 또는 표시
    grid = torchvision.utils.make_grid(torch.cat(images, dim=0), nrow=int(num_images**0.5))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    plt.savefig("sampled_student_images.png")
    print("Sampled images saved to sampled_student_images.png")

if __name__ == "__main__":
    model_weights_path = "./unet_student_final_weights.pth" # 저장된 모델 가중치 경로
    sample_images(model_weights_path, num_images=4)