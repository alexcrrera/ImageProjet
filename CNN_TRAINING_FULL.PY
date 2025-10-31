import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
data_dir = Path("assets/Data/Video_V/dataset_3class")  # <-- your dataset folder
num_classes = 3
batch_size = 64
num_epochs_stage1 = 15       # frozen backbone
num_epochs_stage2 = 15       # fine-tuning
learning_rate_stage1 = 1e-3
learning_rate_stage2 = 1e-4
val_split = 0.2
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# =========================================================
# MODEL DEFINITION
# =========================================================
def make_model(trainable_backbone=False):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    if not trainable_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model.to(device)

criterion = nn.CrossEntropyLoss()

# =========================================================
# TRAINING FUNCTION
# =========================================================
def train_model(model, optimizer, scheduler, num_epochs, stage_name,
                train_loader, val_loader, patience, class_names):

    best_acc = 0.0
    wait = 0
    train_losses, val_losses = [], []
    y_true_best, y_pred_best = [], []
    best_model = None

    for epoch in range(num_epochs):
        start = time.time()
        model.train()
        running_loss, correct, total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        train_losses.append(train_loss)

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        y_true, y_pred = [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_losses.append(val_loss)
        scheduler.step()

        print(f"{stage_name} Epoch [{epoch+1:02d}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"Time: {time.time()-start:.1f}s")

        # Early stopping
        if val_acc > best_acc + 1e-4:
            best_acc = val_acc
            best_model = model.state_dict().copy()
            y_true_best, y_pred_best = y_true, y_pred
            torch.save(best_model, f"best_mobilenetv2_{stage_name.lower()}.pth")
            print(f"New best model saved ({best_acc*100:.2f}% val acc)")
            wait = 0
        else:
            wait += 1
            print(f"No improvement ({wait}/{patience})")
            if wait >= patience:
                print(f"\nEarly stopping after {patience} epochs.")
                break

    return model, best_acc, y_true_best, y_pred_best, train_losses, val_losses


# =========================================================
# MAIN EXECUTION
# =========================================================
if __name__ == "__main__":
    print(f"✅ Using device: {device}")

    # =========================================================
    # TRANSFORMS
    # =========================================================
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(8),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # =========================================================
    # DATASET LOADING
    # =========================================================
    full_dataset = datasets.ImageFolder(data_dir)
    train_len = int((1 - val_split) * len(full_dataset))
    val_len = len(full_dataset) - train_len
    train_ds, val_ds = random_split(full_dataset, [train_len, val_len])

    train_ds.dataset.transform = train_tf
    val_ds.dataset.transform = val_tf

    # Balanced sampling
    train_targets = [full_dataset.samples[i][1] for i in train_ds.indices]
    class_counts = np.bincount(train_targets)
    class_weights = 1. / class_counts
    sample_weights = [class_weights[t] for t in train_targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    num_workers = max(1, os.cpu_count() // 2)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    class_names = full_dataset.classes
    print("Classes:", class_names)
    print("Class distribution (train):", dict(zip(class_names, class_counts)))

    # =========================================================
    # STAGE 1 — Train classifier only
    # =========================================================
    model = make_model(trainable_backbone=False)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate_stage1)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model, acc_stage1, y_true_stage1, y_pred_stage1, train_losses, val_losses = train_model(
        model, optimizer, scheduler, num_epochs_stage1, "Stage1",
        train_loader, val_loader, patience, class_names
    )

    # =========================================================
    # STAGE 2 — Fine-tune last 3 bottleneck blocks
    # =========================================================
    print("\nUnfreezing last 3 bottleneck blocks for fine-tuning...")
    for name, param in list(model.features.named_parameters())[-3*6:]:
        param.requires_grad = True

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate_stage2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model, acc_stage2, y_true, y_pred, train_losses2, val_losses2 = train_model(
        model, optimizer, scheduler, num_epochs_stage2, "Stage2",
        train_loader, val_loader, patience, class_names
    )

    # =========================================================
    # EVALUATION
    # =========================================================
    final_acc = max(acc_stage1, acc_stage2)
    print(f"\nTraining complete. Best validation accuracy: {final_acc*100:.2f}%")
    torch.save(model.state_dict(), "mobilenetv2_bird_drone_bg_final2.pth")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix (Validation Set)")
    plt.show()

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    