from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import mnist_reader

# 載入資料
x_train, y_train = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='t10k')

# Subset selection: 中心裁切 14x14
def center_crop_14x14(flat_images):
    images = flat_images.reshape(-1, 28, 28)
    cropped = images[:, 7:21, 7:21]  # 中心區域
    return cropped.reshape(-1, 14 * 14)

x_train_subset = center_crop_14x14(x_train)
x_test_subset = center_crop_14x14(x_test)

# 輸出資料夾
output_dir = "subset_gnb_output"
os.makedirs(output_dir, exist_ok=True)

# 訓練 GNB 模型
gnb = GaussianNB()
gnb.fit(x_train_subset, y_train)
y_pred = gnb.predict(x_test_subset)

# 評估結果
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy (Subset 14x14) + GNB: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 畫出混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title("Confusion Matrix: Subset (14x14) + GNB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_subset_gnb.png"))
plt.close()

# 儲存 classification report
report = classification_report(y_test, y_pred, output_dict=True)
flat_report = {}
for label, scores in report.items():
    if isinstance(scores, dict):
        for metric, value in scores.items():
            flat_report[f"{label}_{metric}"] = value
    else:
        flat_report[label] = scores
flat_report["feature_type"] = "subset_14x14"
flat_report["accuracy"] = accuracy

# 儲存為 CSV
df = pd.DataFrame([flat_report])
df.to_csv(os.path.join(output_dir, "classification_report_subset_gnb.csv"), index=False)
