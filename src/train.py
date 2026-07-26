import torch
import torch.nn as nn
from tqdm import tqdm
import os

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total

def train_model(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    os.makedirs('checkpoints', exist_ok=True)
    best_val_acc = 0

    for epoch in range(num_epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save({'model_state': model.state_dict(), 'val_acc': vl_acc}, 'checkpoints/best_model.pth')

        print(f"Epoch {epoch+1:02d}/{num_epochs} | Train Acc: {tr_acc*100:.2f}% | Val Acc: {vl_acc*100:.2f}%")
        
    print(f" Training is over! Best Accuracy: {best_val_acc*100:.2f}%")