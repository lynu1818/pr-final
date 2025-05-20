from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import mnist_reader

# 資料輸出資料夾
output_dir = "subset_svm_output"
os.makedirs(output_dir, exist_ok=True)

# === 1. 資料讀取與 subset selection ===

# 讀取 Oracle MNIST 資料
x_train, y_train = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='t10k')

# 取中心的 14x14 區域並 flatten 成 196 維
def center_crop_14x14(flat_images):
    images = flat_images.reshape(-1, 28, 28)
    cropped = images[:, 7:21, 7:21]
    return cropped.reshape(-1, 14 * 14)

x_train_center = center_crop_14x14(x_train)
x_test_center = center_crop_14x14(x_test)

# === 2. 使用 SVM 直接分類（無降維） ===

clf = SVC(kernel='rbf', gamma='scale', probability=True)
clf.fit(x_train_center, y_train)
y_pred = clf.predict(x_test_center)

# === 3. 評估與報告輸出 ===

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print(f"✅ Accuracy (Subset Selection + SVM): {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 混淆矩陣圖
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix: Subset (Center 14x14) + SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_subset_svm.png"))
plt.close()

# 儲存報告為 CSV
flat_report = {}
for label, scores in report.items():
    if isinstance(scores, dict):
        for metric, value in scores.items():
            flat_report[f"{label}_{metric}"] = value
    else:
        flat_report[label] = scores
flat_report["feature_type"] = "subset_14x14"
flat_report["accuracy"] = accuracy

df = pd.DataFrame([flat_report])
df.to_csv(os.path.join(output_dir, "classification_report_subset_svm.csv"), index=False)
