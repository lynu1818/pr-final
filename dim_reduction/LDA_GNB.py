from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import mnist_reader

output_dir = "lda_gnb_output"
os.makedirs(output_dir, exist_ok=True)

# 載入資料
x_train, y_train = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='t10k')

# 攤平成向量
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# LDA 可降至 n_classes - 1 維（這裡是 9）
lda_components = 9

# 執行 LDA
lda = LinearDiscriminantAnalysis(n_components=lda_components)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# GNB 訓練
gnb = GaussianNB()
gnb.fit(x_train_lda, y_train)
y_pred = gnb.predict(x_test_lda)

# 評估
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
print(f"✅ Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title("Confusion Matrix: LDA + GNB")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "confusion_matrix_lda_gnb.png"))
plt.close()

# 儲存報告
flat_report = {}
for label, scores in report.items():
    if isinstance(scores, dict):
        for metric, value in scores.items():
            flat_report[f"{label}_{metric}"] = value
    else:
        flat_report[label] = scores
flat_report["lda_components"] = lda_components
flat_report["accuracy"] = accuracy

# 儲存為 CSV
df = pd.DataFrame([flat_report])
df.to_csv(os.path.join(output_dir, "classification_report_lda_gnb.csv"), index=False)
