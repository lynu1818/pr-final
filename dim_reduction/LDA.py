from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import mnist_reader

x_train, y_train = mnist_reader.load_data('data/oracle', kind='train')
x_test, y_test = mnist_reader.load_data('data/oracle', kind='t10k')

# 1. LDA 降維：最多只能降到 (n_classes - 1) 維，這裡是 9 維
lda = LinearDiscriminantAnalysis(n_components=9)
x_train_lda = lda.fit_transform(x_train, y_train)
x_test_lda = lda.transform(x_test)

# 2. 使用 Gaussian Naive Bayes 進行分類
gnb = GaussianNB()
gnb.fit(x_train_lda, y_train)
y_pred = gnb.predict(x_test_lda)

# 3. 評估結果
accuracy = accuracy_score(y_test, y_pred)
print("✅ Accuracy on Oracle-MNIST with LDA + GNB:", accuracy)

# 4. 顯示分類報告
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 5. 畫出混淆矩陣
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: LDA + GNB")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
