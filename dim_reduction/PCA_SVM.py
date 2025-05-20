from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import mnist_reader
import seaborn as sns
import os
import pandas as pd

output_dir = "pca_svm_output"
os.makedirs(output_dir, exist_ok=True)

# 載入資料
x_train, y_train = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('/home/lynuc/pr-final/oracle-mnist/data/oracle', kind='t10k')

# 攤平成向量
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))

# 不進行標準化
x_train_scaled = x_train
x_test_scaled = x_test

# 要測試的 PCA 組件數
components_list = [2, 4, 8, 16, 32]
results = []

for n_components in components_list:
    print(f"\n=== PCA with {n_components} components ===")

    # 執行 PCA
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train_scaled)
    x_test_pca = pca.transform(x_test_scaled)

    # SVM 訓練
    clf = SVC(kernel='rbf', gamma='scale', probability=True)
    clf.fit(x_train_pca, y_train)
    y_pred = clf.predict(x_test_pca)

    # 評估
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # 儲存文字結果
    flat_report = {}
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                flat_report[f"{label}_{metric}"] = value
        else:
            flat_report[label] = scores  # accuracy

    flat_report["pca_components"] = n_components
    flat_report["accuracy"] = acc
    results.append(flat_report)
    
        # 儲存 confusion matrix 圖片
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=np.unique(y_test),
                yticklabels=np.unique(y_test))
    plt.title(f"Confusion Matrix: PCA={n_components}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_pca{n_components}.png"))
    plt.close()


    # 額外視覺化僅做 2 維 PCA
    if n_components == 2:
        # PCA 投影圖（按原始 label 上色）
        plt.figure(figsize=(8, 6))
        for digit in np.unique(y_train):
            idx = y_train == digit
            plt.scatter(x_train_pca[idx, 0], x_train_pca[idx, 1], label=str(digit), alpha=0.5)
        plt.title("PCA with 2 Components")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pca2_visualization.png"))
        plt.close()

        # SVM 預測結果視覺化
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            x_train_pca[:, 0], x_train_pca[:, 1],
            c=clf.predict(x_train_pca), cmap='tab10', s=5, alpha=0.6
        )
        plt.title("SVM Predictions in PCA 2D Space")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(scatter, label="Predicted Label")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "svm_prediction_scatter_pca2.png"))
        plt.close()

# 儲存整體結果
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "classification_report_all_pca_svm.csv"), index=False)
