# 🍷 Red Wine Quality Prediction

## 📌 Overview
This project aims to predict the quality of red wine using a deep learning model built with **TensorFlow/Keras**. The model is trained on the **Red Wine Quality dataset** and optimized using **batch normalization**, **dropout regularization**, and **early stopping**.

## 📂 Dataset
The dataset used in this project is **"Red Wine Quality"** from **Cortez et al., 2009**. It includes 11 physicochemical attributes of red wine and a target variable:
- **Fixed Acidity**
- **Volatile Acidity**
- **Citric Acid**
- **Residual Sugar**
- **Chlorides**
- **Free Sulfur Dioxide**
- **Total Sulfur Dioxide**
- **Density**
- **pH**
- **Sulphates**
- **Alcohol**
- **Quality (Target Variable: Integer score from 3 to 8)**

## 🛠️ Installation & Requirements
To run this project, install the following dependencies:
```bash
pip install pandas tensorflow
```

## 🚀 Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/red-wine-quality.git
cd red-wine-quality
```

### 2️⃣ Run the Model
Ensure the dataset is located at the correct path and execute:
```bash
python red_wine_model.py
```

## 🔄 Data Preprocessing
1. **Splitting the dataset**
   - 70% Training Data
   - 30% Validation Data
2. **Normalization**
   - Min-max scaling applied to all features to normalize values between 0 and 1.
3. **Feature Selection**
   - The target variable (`quality`) is separated from the feature set.

## 📊 Model Architecture
The model is a **fully connected neural network** with the following layers:
- **Dense (64 neurons, ReLU activation, Input Shape: 11 features)**
- **Batch Normalization**
- **Dropout (0.5)**
- **Dense (32 neurons, ReLU activation)**
- **Batch Normalization**
- **Dropout (0.3)**
- **Dense (32 neurons, ReLU activation)**
- **Batch Normalization**
- **Output Layer (1 neuron, regression output for quality score)**

### 🔧 Model Compilation
- **Optimizer:** Adam
- **Loss Function:** Mean Absolute Error (MAE)
- **Batch Size:** 256
- **Epochs:** 500 (with Early Stopping)
- **Early Stopping:** Stops training if validation loss doesn't improve for 20 epochs.

## 📈 Model Training
The model was trained for **152 epochs** before early stopping, achieving:
- **Final Training Loss:** 0.1132
- **Final Validation Loss:** 0.1018

## 📜 License
This project is open-source under the **MIT License**.

---
Feel free to fork and improve! 🚀
