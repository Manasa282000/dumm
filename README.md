# 🧪 EA_Assignment3

This repository contains solutions for an assignment focusing on advanced neural network architectures. It includes implementations for:
1.  **Physics-Informed Neural Network (PINN)** for solving the 2D Eikonal equation.
2.  **Neural Ordinary Differential Equation (Neural ODE)** for 2D classification, compared against a standard neural network.

The Python script `EA_Assignment3.py` includes solutions for both Question 1 and Question 2.

## Table of Contents
1.  [Question 1: PINN for 2D Eikonal Equation](#question-1--physics-informed-neural-network-pinn-for-solving-the-2d-eikonal-equation)
    *   [Overview](#-overview-q1)
    *   [Components](#-components)
    *   [Results (Q1)](#-results-q1)
    *   [Observation (Q1)](#-observation-q1)
2.  [Question 2: Neural ODE vs Standard Neural Network for 2D Classification](#question-2--neural-ode-vs-standard-neural-network-for-2d-classification)
    *   [Overview](#-overview-q2)
    *   [Implementation Details](#-implementation-details)
    *   [Model Architectures](#-model-architectures)
    *   [Results (Q2)](#-results-q2)
3.  [🚀 Running the Code](#-running-the-code)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Execution](#execution)
4.  [📝 Additional Notes](#-additional-notes)

## Question 1: 🧠 Physics-Informed Neural Network (PINN) for Solving the 2D Eikonal Equation

This project compares two neural network models—a data-only model and a Physics-Informed Neural Network (PINN)—to approximate the activation time field ( $T(x, y)$ ) in a 2D domain.
The PINN model incorporates the Eikonal equation as a soft constraint:

$$ |\nabla T(x, y)| \cdot V(x, y) = 1 $$

### 📌 Overview (Q1)

*   **Objective**: Approximate the scalar field ( $T(x, y)$ ), representing activation time.
*   **Models**:
    *   **Data-only Neural Network**: Trained using Mean Squared Error (MSE) loss only.
    *   **Physics-Informed Neural Network (PINN)**: Trained using MSE + physics residual loss.

### 📋 Components

🔹 **Data Generation**
*   Synthetic generation of:
    *   True activation time ( $T(x, y)$ )
    *   Conduction velocity ( $V(x, y)$ )
*   Uses Latin Hypercube Sampling (LHS) for effective coverage.

🔹 **Neural Network**
*   Fully connected architecture (EikonalNet) with:
    *   Configurable hidden layers and neurons
    *   Tanh activations
*   Uses PyTorch `autograd` for Eikonal residuals.

🔹 **Training Loop**
*   **Optimizer**: Adam
*   **Loss Functions**:
    *   Data-only: MSE
    *   PINN: MSE + Weighted Eikonal Residual

🔹 **Evaluation & Visualization**
*   Contour plots: Ground truth & predictions
*   Error maps and loss curves
*   **Metric**: Root Mean Squared Error (RMSE)

### 📊 Results (Q1)

▶️ **With 50 Training Points**

| Model                 | RMSE   |
| :-------------------- | :----- |
| Data-only Model       | 0.0224 |
| Physics-Informed NN   | 0.0370 |

▶️ **With 30 Training Points**

| Model                 | RMSE   |
| :-------------------- | :----- |
| Data-only Model       | 0.0540 |
| Physics-Informed NN   | 0.1301 |

### 📌 Observation (Q1)
The data-only model performed better, especially with fewer points, likely due to difficulties in estimating gradients under sparse data conditions for the PINN.

---

## Question 2: 🧠 Neural ODE vs Standard Neural Network for 2D Classification

This experiment compares a standard feedforward neural network with a Neural Ordinary Differential Equation (Neural ODE) model on synthetic 2D classification tasks.

### 📌 Overview (Q2)

*   **Objective**: Classify 2D points using discrete and continuous-depth models.
*   **Models**:
    *   **Standard Neural Network** (ReLU activations)
    *   **Neural ODE** (ODE solver + adjoint method)
*   **Datasets**:
    *   Default: `make_blobs`
    *   Optional: `make_moons`, `make_circles` from `sklearn.datasets`

### 🔧 Implementation Details

*   **Standardization**: Input normalized via `StandardScaler`
*   **Loss**: `CrossEntropyLoss`
*   **Optimizer**: Adam, learning rate = 0.01
*   **Epochs**: 500
*   **Device**: GPU (CUDA) if available

### 🧪 Model Architectures

🔹 **Standard Neural Network**
1.  Input Layer
2.  Linear Layer
3.  ReLU Activation
4.  Linear Layer
5.  Output Layer

🔸 **Neural ODE**
1.  Input Layer
2.  Linear Layer
3.  **ODE Solver Block**: Learns a continuous transformation governed by $\frac{dh}{dt} = f(h(t), t)$
4.  Linear Layer
5.  Output Layer

Efficient backpropagation for the ODE block is achieved via `torchdiffeq.odeint_adjoint`.

### 📊 Results (Q2)

| Model         | Training Accuracy | Test Accuracy |
| :------------ | :---------------- | :------------ |
| Standard NN   | 100.00%           | 100.00%       |
| Neural ODE    | 100.00%           | 100.00%       |

---

## 🚀 Running the Code

### Prerequisites
*   Python 3.x
*   pip

### Installation
Clone the repository (if applicable) and install the required dependencies:
```bash
pip install -r requirements.txt
