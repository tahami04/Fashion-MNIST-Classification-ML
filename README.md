# Fashion-MNIST-Classification-ML
A machine learning project comparing K-Nearest Neighbors (KNN), Decision Trees, and Support Vector Machines (SVM) for classifying Fashion MNIST images. Includes data preprocessing, model evaluation, confusion matrix visualizations, and performance analysis.
# Fashion MNIST Classification - Machine Learning Comparison

## Project Overview
This repository contains code for training and evaluating three machine learning models (KNN, Decision Tree, SVM) on the Fashion MNIST dataset. The goal is to classify grayscale images of 10 fashion categories (e.g., T-shirts, shoes) and compare model performance using accuracy, classification reports, and confusion matrices.

### Key Features:
- Data preprocessing (flattening, normalization).
- Implementation of KNN, Decision Tree, and SVM classifiers.
- Visualization of confusion matrices using Seaborn.
- Performance comparison across models.

## Repository Structure
- `fashion_mnist_classification.ipynb`: Main Python script for data processing, model training, and evaluation.
- `README.md`: This documentation.

## Code Overview

### 1. Dataset Preparation
- Loads the Fashion MNIST dataset using Keras.
- Flattens 28x28 images into 1D arrays (784 features).
- Normalizes pixel values to [0, 1].
- Splits training data into 80% training and 20% validation sets.

### 2. Models Implemented
1. **K-Nearest Neighbors (KNN)**:  
   - Accuracy: **85.28%**  
   - Best F1-scores: Class 1 (Trouser, 0.98), Class 9 (Ankle Boot, 0.93).  
   - Struggled with Class 6 (Shirt, F1=0.61).

2. **Decision Tree**:  
   - Accuracy: **80.58%**  
   - Limited depth (`max_depth=10`) to prevent overfitting.  
   - Poor performance on Class 6 (Shirt, F1=0.54).

3. **Support Vector Machine (SVM)**:  
   - Accuracy: **85.38%** (Best overall)  
   - Linear kernel used for efficiency.  
   - Strong performance on Class 1 (Trouser, F1=0.97) and Class 9 (Ankle Boot, F1=0.95).

### 3. Key Results
| Model          | Accuracy | Worst Class (F1-Score) | Best Class (F1-Score) |
|----------------|----------|------------------------|-----------------------|
| KNN            | 85.28%   | Class 6 (0.61)         | Class 1 (0.98)        |
| Decision Tree  | 80.58%   | Class 6 (0.54)         | Class 8 (0.93)        |
| SVM            | 85.38%   | Class 6 (0.62)         | Class 9 (0.95)        |

**Visualizations**: Confusion matrices for each model highlight misclassifications (e.g., Shirts/Class 6 confused with T-shirts/Class 0).

## How to Use
1. **Dependencies**:  
   ```bash
   pip install numpy matplotlib scikit-learn seaborn tensorflow
Run the script:

bash
Copy
python fashion_mnist_classification.ipynb
This will train models, print metrics, and display interactive confusion matrices.

Results Interpretation
Class 6 (Shirt): Consistently the hardest to classify across all models.

SVM vs. KNN: SVM marginally outperforms KNN but requires more computational resources.

Decision Tree Trade-off: Simpler and faster but less accurate.

Future Improvements
Experiment with CNN architectures for image-specific feature extraction.

Hyperparameter tuning (e.g., SVM kernels, KNN neighbors, tree depth).

Add real-time image prediction examples.

Address Class 6 challenges using data augmentation.

License
MIT License. Dataset sourced from Keras/TensorFlow.
