import torch
import CD.teacher as teacher
import CD.student as student

from dataset import get_dataloader

DEVICE = teacher.DEVICE
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
EPOCHS = 100

dataloader = get_dataloader(batch_size=BATCH_SIZE)

model = student.UNetStudent().to(DEVICE)
model.train()

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

for epoch in range(EPOCHS):
    for step, batch in enumerate(dataloader):
        x0 = batch[0].to(DEVICE)

        with torch.no_grad():
            xt1, t1, xt2, t2 = teacher.generate_consistency_pair(x0)
            target = teacher.unet(xt1, t1).sample
        
        pred = model(xt2, t2)
        loss = criterion(pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch+1} / step {step}: Loss = {loss.item():.6f}")