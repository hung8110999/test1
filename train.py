import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# ======================
# Force dùng GPU 0
# ======================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ======================
# Load dataset
# ======================
df = pd.read_csv("/home/hung.nguyen.e/MLOps/dataset.csv", encoding="utf-8")

X = df[["x1", "x2"]].values
y = df["y"].values

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

# ======================
# Perceptron Model
# ======================
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

model = Perceptron().to(device)

# ======================
# Loss & Optimizer
# ======================
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# ======================
# Train
# ======================
epochs = 50

for epoch in range(epochs):
    optimizer.zero_grad()
    
    outputs = model(X)
    loss = criterion(outputs, y)
    
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# ======================
# Save model
# ======================
torch.save(model.state_dict(), "perceptron_gpu0.pt")
print("Model saved: perceptron_gpu0.pt")