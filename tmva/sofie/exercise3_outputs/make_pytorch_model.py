import torch
import torch.nn as nn

model = nn.Sequential(
           nn.Linear(32,16),
           nn.ReLU(),
           nn.Linear(16,8),
           nn.ReLU()
           )

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.01)

x=torch.randn(2,32)
y=torch.randn(2,8)

for i in range(500):
    y_pred = model(x)
    loss = criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

model.eval()
m = torch.jit.script(model)
torch.jit.save(m,'PyTorchModel.pt')

