# EMNIST Character Recognition using Deep Learning
The project aims to classify characters from the EMNIST Balanced dataset using deep learning. Objectives include data preparation, developing MLP and CNN models, hyperparameter tuning, and performance evaluation. 
## Project Overview
The goal of this project is to develop and optimize deep learning models to accurately recognize characters from the Extended MNIST (EMNIST) dataset, specifically the Balanced subset. The project involves creating and fine-tuning both Multi-Layer Perceptron (MLP) and Convolutional Neural Network (CNN) models to achieve high accuracy in character recognition.

## Objectives
Data Preparation: Load, preprocess, and visualize the EMNIST Balanced dataset.
Model Development: Build baseline MLP and CNN models.
Hyperparameter Optimization: Fine-tune hyperparameters to improve model performance.
Evaluation: Evaluate the models on a test dataset and compare their performance.
Visualization: Visualize the results, including training curves, confusion matrices, and sample predictions.

## Libraries
Essential libraries for data processing, visualization, model building, and evaluation are imported.
```python
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
import tensorflow as tf
from keras import regularizers
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization, Dropout, Activation
from keras.optimizers import SGD, Adam, Adagrad
from keras.optimizers.schedules import ExponentialDecay, InverseTimeDecay
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
```
# Dataset
EMNIST Balanced dataset: https://www.kaggle.com/datasets/crawford/emnist

# Key Results
- CNN Outperformance: CNN consistently outperformed MLP in accuracy and loss metrics.
- Accuracy: Higher testing accuracy for CNN. 
- Loss: Lower testing loss for CNN.
- Confusion Matrix Analysis: Both models struggled with similar-looking characters (e.g., 0, o, O and 1, l, L), but CNN showed better precision.
- Evaluation Metrics: CNN had superior precision, recall, and F1-score.
- Feature Extraction: CNNs excel at automatic feature extraction from images.
- Hierarchical Representation: CNNs capture complex relationships effectively through layered representations.
- Translation Invariance: CNNs are robust to transformations such as translations and rotations.
- Parameter Efficiency: CNNs are more parameter-efficient than MLPs for high-dimensional data.
- Performance Improvement: CNN's task-specific architecture leads to enhanced performance.
