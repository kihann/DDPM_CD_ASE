import torch
import torch.nn.functional as F

from models import teacher, student
from dataset import get_dataloader

DEVICE = teacher.DEVICE
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 10

dataloader = get_dataloader(batch_size=BATCH_SIZE)

model = student.UNetStudent(diffusion_steps=1000, skip_exponent=2.5).to(DEVICE)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
        x0 = batch[0].to(DEVICE)

        with torch.no_grad():
            xt1, t1, xt2, t2 = teacher.generate_consistency_pair(x0)
            target = teacher.unet(xt1, t1).sample
        
        pred_list = model(xt2, t2, return_all=True)

        lambdas = [0.1, 0.15, 0.2, 0.25, 0.3]
        loss = 0.0
        for i, pred in enumerate(pred_list):
            loss += lambdas[i] * criterion(pred, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch+1} / step {step+1}: Loss = {loss.item():.6f}")

# 학습 완료 후 모델 저장
save_path = "./unet_student_final_weights.pth" # 또는 원하는 경로와 파일명
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")