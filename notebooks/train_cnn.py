import torch
from torch import nn, optim
from dataloaders import train_loader, val_loader, test_loader
from dataloader_randomized import randomized_train_loader
#from models import PneumoniaResNet
from models import PneumoniaCNN
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

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
    f1 = f1_score(all_labels, all_preds, average='binary')  # Assuming binary classification with labels 0 and 1

    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    return accuracy, f1




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PneumoniaCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the normal model
    train_net(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)
    #train_net(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)
    #model.load_state_dict(torch.load("../model_state_dicts/cnn_model_transformed.pt", map_location=device))
    model.eval()

    # Evaluate the model
    # print("Validation Set:")
    # eval_net(model, val_loader, device)
    print("Test Set for Normal Training:")
    eval_net(model, test_loader, device)



    torch.save(model.state_dict(), "../model_state_dicts/cnn_model_with_sampling_1.pt")


    model = PneumoniaCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the randomized model
    train_net(model, randomized_train_loader, val_loader, criterion, optimizer, device, epochs=5)
    # train_net(model, train_loader, val_loader, criterion, optimizer, device, epochs=5)
    # model.load_state_dict(torch.load("../model_state_dicts/cnn_model_transformed.pt", map_location=device))
    model.eval()

    # Evaluate the model
    # print("Validation Set:")
    # eval_net(model, val_loader, device)
    print("Test Set Randomized Training:")
    eval_net(model, test_loader, device)

    torch.save(model.state_dict(), "../model_state_dicts/cnn_model_randomized_with_sampling_1.pt")