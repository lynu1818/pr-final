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

# 標準化資料
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


def visualize_pca(x, y, pca_components, filename):
    pca = PCA(n_components=pca_components)
    x_pca = pca.fit_transform(x)

    if pca_components == 2:
        plt.figure(figsize=(8, 6))
        for digit in np.unique(y):
            idx = y == digit
            plt.scatter(x_pca[idx, 0], x_pca[idx, 1], label=str(digit), alpha=0.5)
        plt.title("PCA with 2 Components")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    return pca, x_pca

def plot_decision_boundary(clf, X, y, title, filename):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                         np.linspace(y_min, y_max, 300))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='tab10')
    for digit in np.unique(y):
        idx = y == digit
        plt.scatter(X[idx, 0], X[idx, 1], label=str(digit), s=15)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig(filename)
    plt.close()


components_list = [2, 4, 8]

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

    # 儲存 confusion matrix 圖片
    plot_confusion_matrix(y_test, y_pred,
        title=f"Confusion Matrix (PCA={n_components})",
        filename=os.path.join(output_dir, f"confusion_matrix_pca{n_components}.png")
    )

    # 儲存文字結果
    flat_report = {}
    for label, scores in report.items():
        if isinstance(scores, dict):
            for metric, value in scores.items():
                flat_report[f"{label}_{metric}"] = value
        else:
            flat_report[label] = scores  # for 'accuracy'

    flat_report["pca_components"] = n_components
    flat_report["accuracy"] = acc  # 雖然 accuracy 已在上面，但可以覆蓋一次
    results.append(flat_report)


    if n_components == 2:
        # 儲存 PCA 資料點圖
        visualize_pca(x_train_scaled, y_train, pca_components=2,
            filename=os.path.join(output_dir, "pca2_visualization.png"))

        # 儲存 decision boundary 圖
        plot_decision_boundary(clf, x_train_pca, y_train,
            title="SVM Decision Boundary (PCA=2)",
            filename=os.path.join(output_dir, "decision_boundary_pca2.png")
        )

# 儲存整體 csv 結果
df = pd.DataFrame(results)
df.to_csv(os.path.join(output_dir, "classification_report.csv"), index=False)
