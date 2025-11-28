# Basics of Neural Networks — Wrocław University of Science and Technology

This repository was created for the **Neural Network Basics** course at the Wrocław University of Science and Technology (PWr).  
First project contains a fully custom implementation of a **Multilayer Perceptron (MLP)** written from scratch in NumPy, along with preprocessing, training pipeline, and visualizations performed on the _diabetes classification dataset_.

---

## 🎯 Project Objective

The main goal of the project was to implement a complete neural network manually — without deep-learning frameworks — and train it on a real medical dataset. The project includes:

- a standalone neural-network implementation (`nn.py`),
- preprocessing and data preparation in Jupyter Notebook,
- train/validation split and model evaluation,
- analysis of training performance (loss, accuracy, weight norms),
- visualizations and performance metrics.

---

# 🧠 Neural Network Implementation (`nn.py`)

The core of this project is a manually implemented MLP model available in:

> **`nn.py`**

This implementation was written entirely in NumPy and reproduces many mechanisms known from modern deep-learning frameworks.

### ✔️ **Network Architecture**

- **Input layer:** 8 features (from diabetes dataset)
- **Hidden layer:** configurable number of neurons
- **Activation functions:** ReLU or Sigmoid
- **Output layer:** 1 neuron
- **Output activation:** Sigmoid — produces probability for binary classification

---

## ⚙️ Key Features of the Implementation

This is not a minimal MLP — it contains several advanced mechanisms:

### 🔧 **1. Custom Adam Optimizer**

- first- and second-moment updates,
- bias correction,
- separate state for all weights and biases.

### 🧨 **2. Dropout Support**

- applied on the hidden layer during training,
- proper scaling of activations.

### 🔒 **3. L2 Regularization**

- configurable strength (`l2_lambda`),
- added directly to gradient calculations.

### 📉 **4. Adaptive Learning Rate (LR Decay)**

- constant decay,
- step-based decay,
- dictionary-based decay configuration.

### 🛑 **5. Early Stopping**

- configurable patience,
- monitored on validation loss,
- automatic restoration of the best weights.

### 📊 **6. Internal Training History Logging**

The model stores:

- full epoch loss,
- batch loss,
- classification error,
- training/validation accuracy,
- validation loss,
- learning rate per epoch,
- weight norms for both layers.

### 📦 **7. Mini-Batch Training**

- configurable batch size,
- shuffled batches per epoch.

### 🔮 **8. Prediction API**

- `predict_proba(X)` → returns probabilities
- `predict(X)` → returns 0/1 labels (default threshold = 0.5)

---

# 📊 Learning Task: Diabetes Classification

The model is trained on the **Pima Indians Diabetes Database**:

This is a binary classification problem predicting whether a patient is likely to have type-2 diabetes based on medical features such as:

- pregnancies,
- glucose concentration,
- blood pressure,
- skin fold thickness,
- insulin level,
- BMI,
- diabetes pedigree function,
- age.

**Target:**

- `1` — diabetes
- `0` — no diabetes

---

# 🧹 Data Preprocessing (Jupyter Notebook)

All preprocessing steps are implemented inside the Jupyter notebook:

**`simplified_nn.ipynb`**

This includes:

- loading the dataset,
- checking for missing values,
- repleacing missing values with medians
- feature scaling / normalization,
- conversion to NumPy arrays,
- splitting into:
  - **training set**
  - **validation set**
  - **test set**

The processed arrays serve as inputs to the MLP implemented in `nn.py`.

---

# 🏋️ Training Procedure

The training loop includes:

- forward propagation,
- MSE loss computation,
- accuracy computation,
- backpropagation,
- Adam optimizer update,
- optional L2 regularization & dropout,
- validation evaluation each epoch,
- early stopping (if enabled).

### 🔍 Metrics Tracked Every Epoch

- training loss (MSE),
- validation loss,
- training accuracy,
- validation accuracy,
- classification error,
- learning rate,
- L2 weight norms.

---

# 📈 Visualizations

The notebook produces standard ML training plots:

- **Loss over epochs** (train & validation),
- **Accuracy over epochs** (train & validation),
- **Classification error**,
- **Weight norms** for both layers,
- **Learning rate decay** (if used).

These visualizations help analyze convergence and detect overfitting.
