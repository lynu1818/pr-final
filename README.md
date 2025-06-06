# Pattern Recognition Final Project

This project explores classification and clustering techniques on two symbolic image datasets: **Oracle Bone Script** and **Egyptian Hieroglyphs**. We apply both **traditional machine learning** and **deep learning** approaches, along with **dimensionality reduction**, to extract and analyze meaningful features.

---

## Project Structure


---

## Methodology Summary

### Oracle Bone Script (`oracle/`)

#### Feature Types:
- Raw 28×28 images
- CNN-extracted features (from `simple_CNN.py`)

#### Dimensionality Reduction:
- **Subset Selection**: 14×14 cropped region
- **PCA**: n ∈ {2, 4, 8, 16, 32}
- **LDA**: n = 9

#### Classifiers:
- SVM with RBF kernel
- Gaussian Naive Bayes (GNB)

#### Clustering:
- K-means
- GMM (via EM algorithm)

#### Additional Models:
- MLP trained directly on raw 28×28 images

---

### Egyptian Hieroglyphs (`egyptian/`)

#### Workflow:
- CNN feature extraction on labeled hieroglyph images
- Same classification and clustering as Oracle dataset

#### Outputs:
- Training curves, accuracy/loss plots
- Confusion matrix, classification report
- Saved weights (`.pth`), label maps (`.json`)

---

## Implementation Tools

- Python 3.x
- Jupyter Notebook
- PyTorch for CNN and MLP models
- scikit-learn for PCA, LDA, SVM, GNB, K-means, and GMM
- Matplotlib / Seaborn for visualization

---

## Key Findings

- CNN-extracted features significantly outperform raw pixel features in classification tasks.
- PCA and LDA provide interpretable low-dimensional spaces for clustering.
- Deep learning models (CNN, MLP) show robustness on symbolic datasets with small sample sizes.

---

## Contributors

- [Your Name / ID]
- [Teammates if applicable]

---

## License

This project is intended for academic research and education purposes only.
