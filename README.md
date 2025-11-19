# Gradient-based Learning

This project implements several classifiers and training routines as part of an assignment on gradient-based learning.  
The goal is to build a log-linear classifier, a multi-layer perceptron (MLP), and an arbitrary-depth MLP, and to evaluate them on several tasks such as language identification and XOR.

---

## Project Structure

- **grad_check.py** – Implements gradient checking to verify correctness of gradient calculations.
- **loglinear.py** – Implementation of a log-linear classifier (softmax classifier).
- **train_loglin.py** – Training code for the log-linear classifier using SGD.
- **mlp1.py** – Implementation of a single-hidden-layer MLP.
- **train_mlp1.py** – Training routine for the MLP (similar to train_loglin).
- **mlpn.py** – Implementation of an arbitrary-depth MLP that supports multiple layers.
- **xor_data.py** – Small dataset for evaluating XOR learning.
- **answers.txt** – Written answers to the assignment questions.
- **data/** – Train, dev, and blind test sets for the language identification task.

---

## Tasks Implemented

### **1. Gradient Checking**
Implemented numerical gradient checking to validate analytical gradients.  
Used to verify the correctness of all classifiers (log-linear and MLPs).

### **2. Log-Linear Classifier**
- Implements:  
  \[
  f(x) = \text{softmax}(Wx + b)
  \]
- Trained using SGD.  
- Evaluated on a language identification task using letter bigram features.
- Achieves >80% accuracy on the dev set (expected baseline).

### **3. Predicting on Blind Test Data**
- Generates predictions for the test set.
- Saves results in `test.pred`, one language ID per line.

### **4. Single Hidden Layer MLP**
- Implements:  
  \[
  f(x) = \text{softmax}(U \cdot \tanh(Wx + b) + b')
  \]
- Trained with SGD.
- Evaluated on:
  - Language identification task
  - XOR problem

### **5. Arbitrary-Depth MLP**
- Supports networks of shape:  
  `create_classifier([input_dim, h1, h2, ..., output_dim])`
- General loss, prediction, and gradient functions for N-layer MLPs.

---

## 📈 Experiments

The assignment includes several experiments:

- Compare log-linear vs. MLP performance on language identification.
- Evaluate unigram vs. bigram features.
- Train MLP on XOR and measure iterations to convergence.

Results and explanations are in **answers.txt**.

---

## ▶️ How to Run

```bash
python grad_check.py
python loglinear.py
python train_loglin.py
python mlp1.py
python train_mlp1.py
