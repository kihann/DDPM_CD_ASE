import os
import torch
from torch_ema import ExponentialMovingAverage

from models import teacher, student
from dataset import get_dataloader

# --- 전역 상수 및 설정 ---
CHECKPOINT_SAVE_DIR = "./checkpoints_cd_rigorous" # 체크포인트 저장 폴더
os.makedirs(CHECKPOINT_SAVE_DIR, exist_ok=True)

DEVICE = teacher.DEVICE
LEARNING_RATE = 1e-6      # 사용자 설정값
BATCH_SIZE = 4
EPOCHS = 1               # 사용자 설정값 (실제 학습 시에는 더 늘릴 수 있음)
EMA_DECAY = 0.9999        # 사용자 설정값

# 체크포인트 저장 주기 (예: 매 에포크마다)
CHECKPOINT_SAVE_EPOCH_INTERVAL = 1
# --- 전역 상수 및 설정 끝 ---

dataloader = get_dataloader(batch_size=BATCH_SIZE)

model = student.UNetStudent(
    sigma_min=teacher.SIGMA_MIN,
    sigma_max=teacher.SIGMA_MAX,
    skip_exponent=2.5
).to(DEVICE)

model_ema = student.UNetStudent(
    sigma_min=teacher.SIGMA_MIN,
    sigma_max=teacher.SIGMA_MAX,
    skip_exponent=2.5
).to(DEVICE)
model_ema.load_state_dict(model.state_dict())

ema_handler = ExponentialMovingAverage(model.parameters(), decay=EMA_DECAY)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

print(f"Starting Consistency Distillation (Algorithm 2 style - Rigorous ODE) training...")
print(f"Device: {DEVICE}, LR: {LEARNING_RATE}, Batch Size: {BATCH_SIZE}, EMA Decay: {EMA_DECAY}")
print(f"Sigma Range for CD: [{teacher.SIGMA_MIN}, {teacher.SIGMA_MAX}], N_CD_STEPS: {teacher.N_CD_STEPS}")
print(f"Checkpoints will be saved to: {os.path.abspath(CHECKPOINT_SAVE_DIR)}")

# --- 학습 루프 ---
for epoch in range(EPOCHS):
    model.train()
    epoch_loss_sum = 0.0
    for step, batch_data in enumerate(dataloader):
        if isinstance(batch_data, (list, tuple)):
            x0 = batch_data[0].to(DEVICE)
        else:
            x0 = batch_data.to(DEVICE)
        
        optimizer.zero_grad()

        xt_online_input, sigma_online_input, xt_target_input, sigma_target_input = \
            teacher.generate_consistency_distillation_pair_rigorous(
                x0, teacher.unet, teacher.scheduler
            )

        online_epsilon_preds_list = model(xt_online_input, sigma_online_input, return_all=True)

        with torch.no_grad():

            current_ema_params = {}
            for name, param in model.named_parameters(): # 온라인 모델에서 EMA 파라미터 가져오기
                if name in ema_handler.shadow_params:
                    current_ema_params[name] = ema_handler.shadow_params[name].clone()
                else:
                    current_ema_params[name] = param.data.clone() # 버퍼 등 학습 안되는 파라미터는 그대로 복사
            
            original_model_ema_state = model_ema.state_dict() # 혹시 모를 복원을 위해 저장 (선택사항)
            model_ema.load_state_dict(current_ema_params) # EMA 파라미터 로드
            
            target_epsilon_preds_list_ema = model_ema(xt_target_input, sigma_target_input, return_all=True)
            
            # model_ema.load_state_dict(original_model_ema_state) # 필요시 복원 (보통은 다음 스텝 EMA 업데이트로 덮어쓰여짐)


        lambdas = [0.011656, 0.031685, 0.086129, 0.234121, 0.636399]
        if len(online_epsilon_preds_list) != len(lambdas):
            num_preds = len(online_epsilon_preds_list)
            lambdas = [1.0/num_preds] * num_preds if num_preds > 0 else []

        total_loss = 0.0
        min_len = min(len(online_epsilon_preds_list), len(target_epsilon_preds_list_ema))
        for i in range(min_len):
            online_pred_eps = online_epsilon_preds_list[i]
            target_pred_eps_ema = target_epsilon_preds_list_ema[i]
            total_loss += lambdas[i] * criterion(online_pred_eps, target_pred_eps_ema.detach())
        
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad: # 손실이 유효한 경우에만 역전파
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 그래디언트 클리핑
            optimizer.step()
            ema_handler.update() # 온라인 모델 파라미터 업데이트 후 EMA 업데이트
            epoch_loss_sum += total_loss.item()
        elif isinstance(total_loss, float) and total_loss == 0.0 and min_len == 0: # 예측 리스트가 비어서 손실이 0.0 float인 경우
             pass # 아무것도 안함
        elif not isinstance(total_loss, torch.Tensor) or not total_loss.requires_grad : # 역전파 불가능한 경우 (예: 예측 리스트 비었을 때)
            if min_len > 0 : # 예측은 있었는데 손실계산에 문제가 생긴 경우 경고
                 print(f"Warning: total_loss is not a valid tensor for backward. Loss value: {total_loss}")


        if (step + 1) % 100 == 0:
            loss_val = total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
            print(f"Epoch {epoch+1}/{EPOCHS}, Step {step+1}/{len(dataloader)}, Loss: {loss_val:.6f}")
    
    epoch_avg_loss = epoch_loss_sum / len(dataloader) if len(dataloader) > 0 else 0
    print(f"Epoch {epoch+1} finished. Avg Loss: {epoch_avg_loss:.6f}")

    # --- 에포크 종료 후 주기적 체크포인트 저장 ---
    if (epoch + 1) % CHECKPOINT_SAVE_EPOCH_INTERVAL == 0 or epoch == EPOCHS - 1:
        # EMA 모델의 가중치를 저장하는 것이 일반적 (더 안정적인 성능을 보임)
        ema_checkpoint_path = os.path.join(CHECKPOINT_SAVE_DIR, f"unet_student_ema_epoch_{(epoch+1):03d}.pth")
        
        # model_ema 인스턴스에 현재 EMA shadow 파라미터를 로드하여 저장
        temp_ema_state_dict = {}
        for name, param in model.named_parameters(): # model의 구조를 따름
            if name in ema_handler.shadow_params:
                temp_ema_state_dict[name] = ema_handler.shadow_params[name].clone()
            else: # 버퍼 등 학습되지 않는 파라미터
                temp_ema_state_dict[name] = param.data.clone()
        
        # model_ema 객체에 EMA 가중치를 로드한 후 model_ema.state_dict() 저장
        # 또는 직접 state_dict를 빌드하여 저장
        torch.save(temp_ema_state_dict, ema_checkpoint_path)
        print(f"Saved EMA model checkpoint to {ema_checkpoint_path}")

        # 온라인 모델도 저장
        online_checkpoint_path = os.path.join(CHECKPOINT_SAVE_DIR, f"unet_student_online_epoch_{(epoch+1):03d}.pth")
        torch.save(model.state_dict(), online_checkpoint_path)
        print(f"Saved Online model checkpoint to {online_checkpoint_path}")

print("Training finished.")