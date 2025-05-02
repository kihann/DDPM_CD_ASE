import torch
import CD.teacher as teacher
import CD.student as student

DEVICE = teacher.DEVICE
LEARNING_RATE = 1e-4

student = student.UNetStudent().to(DEVICE)
student.train()

optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.MSELoss()

x0_student = student(teacher.xt, teacher.timestep)

loss = criterion(x0_student, teacher.x0_teacher)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"{loss=}")