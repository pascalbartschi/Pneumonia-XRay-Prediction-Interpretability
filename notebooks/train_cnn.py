import torch
from torch import nn, optim
from sklearn.metrics import accuracy_score
from dataloaders import train_loader, val_loader, test_loader
from models import PneumoniaResNet
from tqdm.notebook import tqdm

def train_net(model, train_loader, val_loader, criterion, optimizer, device, epochs=5):
    model.to(device)
    progress_bar = tqdm(range(epochs), desc="Training Progress", leave=True)
    for epoch in progress_bar:
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        progress_bar.set_postfix(epoch_loss=running_loss / len(train_loader))
        print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader):.4f}")

        # Validation step
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        val_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Validation Accuracy: {val_accuracy:.4f}")

def eval_net(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=True):
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaResNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_net(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)

    # Evaluate the model
    # print("Validation Set:")
    # eval_net(model, val_loader, device)
    print("Test Set:")
    eval_net(model, test_loader, device)