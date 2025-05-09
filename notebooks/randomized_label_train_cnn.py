import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from dataloaders import train_loader, val_loader, test_loader
from models import PneumoniaCNN
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PneumoniaCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Randomize labels
np.random.seed(42)
train_loader.dataset.targets = np.random.permutation(train_loader.dataset.targets).tolist()

# Training loop
for epoch in range(5):  # Feel free to increase for better performance
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

# # Evaluation
# model.eval()
# all_preds = []
# all_labels = []
#
# with torch.no_grad():
#     for inputs, labels in test_loader:
#         inputs = inputs.to(device)
#         outputs = model(inputs)
#         preds = torch.argmax(outputs, dim=1).cpu().numpy()
#         all_preds.extend(preds)
#         all_labels.extend(labels.numpy())
#
# print(f"Test Accuracy: {accuracy_score(all_labels, all_preds):.4f}")

#torch.save(model, "cnn_model.pt")
torch.save(model.state_dict(), "cnn_model_randomized.pt")