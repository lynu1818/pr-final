import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import mnist_reader

# === 1. 資料準備 ===

x_train, y_train = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='t10k')

x_train = x_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

y_train = y_train.astype(np.int64)
y_test = y_test.astype(np.int64)

train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train))
test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

output_dir = "cnn_output_pytorch"
os.makedirs(output_dir, exist_ok=True)


# === 2. CNN 模型定義 ===

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))  # (N, 32, 13, 13)
        x = self.pool2(F.relu(self.conv2(x)))  # (N, 64, 5, 5)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)

# === 3. 損失與優化器 ===

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === 4. 訓練模型 ===

train_losses = []
train_accuracies = []

for epoch in range(20):
    correct = 0
    total = 0
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    # 預測與統計正確數
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = train_loss / len(train_loader)
    train_acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(train_acc)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")


# 儲存最後訓練完的模型
torch.save(model.state_dict(), "best_cnn_model.pt")

# === 4.5 畫出 Loss 與 Accuracy 曲線 ===
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, marker='o', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.xticks(range(1, len(train_losses)+1))
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, marker='o', label='Train Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.xticks(range(1, len(train_accuracies)+1))
plt.grid(True)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "training_curves.png"))
plt.close()


# === 5. 測試與預測 ===

model.load_state_dict(torch.load("best_cnn_model.pt"))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

accuracy = accuracy_score(all_labels, all_preds)
print(f"✅ CNN Accuracy (PyTorch 28x28): {accuracy:.4f}")
print(classification_report(all_labels, all_preds))

# === 6. 輸出結果 ===

output_dir = "cnn_output_pytorch"
os.makedirs(output_dir, exist_ok=True)

# 混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix: CNN (PyTorch) on 28x28")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_cnn_28x28.png"))
plt.close()

# classification_report 儲存
report = classification_report(all_labels, all_preds, output_dict=True)
flat_report = {}
for label, scores in report.items():
    if isinstance(scores, dict):
        for metric, value in scores.items():
            flat_report[f"{label}_{metric}"] = value
    else:
        flat_report[label] = scores
flat_report["feature_type"] = "full_28x28"
flat_report["accuracy"] = accuracy

df = pd.DataFrame([flat_report])
df.to_csv(os.path.join(output_dir, "classification_report_cnn.csv"), index=False)
