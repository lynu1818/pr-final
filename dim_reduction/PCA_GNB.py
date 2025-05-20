from sklearn.decomposition import PCA
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

x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# 輸出資料夾
output_dir = "pca_gnb_output"
os.makedirs(output_dir, exist_ok=True)

# 要測試的 PCA 維度
pca_components_list = [2, 4, 8, 16, 32]

# 儲存每組結果
all_results = []

for pca_components in pca_components_list:
    print(f"\n=== PCA({pca_components}) + GNB ===")

    # 1. 執行 PCA
    pca = PCA(n_components=pca_components)
    x_train_pca = pca.fit_transform(x_train)
    x_test_pca = pca.transform(x_test)

    # 2. Gaussian Naive Bayes 分類
    gnb = GaussianNB()
    gnb.fit(x_train_pca, y_train)
    y_pred = gnb.predict(x_test_pca)

    # 3. 評估結果
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))

    # 4. 畫出混淆矩陣
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix: PCA({pca_components}) + GNB")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_pca{pca_components}_gnb.png"))
    plt.close()

    # 5. 儲存 classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    flat_report = {}
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                flat_report[f"{label}_{metric}"] = value
        else:
            flat_report[label] = scores
    flat_report["pca_components"] = pca_components
    flat_report["accuracy"] = accuracy
    all_results.append(flat_report)

# 儲存所有結果成 CSV
df = pd.DataFrame(all_results)
df.to_csv(os.path.join(output_dir, "classification_report_all_pca_gnb.csv"), index=False)
