# This Cpp Machine Learning

這是一個基本的深度學習框架，包含了多種層、激活函數、損失函數以及矩陣操作。該框架的主要目的是提供一個簡單的方式來建立和訓練神經網絡。

## 架構圖

以下是該框架的程式碼架構圖：

```mermaid
graph TD;
    A[MyFrame] -->|add| B[Layer]
    A -->|train_img_input| C[LossFunc]
    B --> D[ConvLayer]
    B --> E[DropoutLayer]
    B --> F[DenseLayer]
    B --> G[FlattenLayer]
    B --> H[SGD]
    B --> I[MMT]
    B --> J[ActiveFunc]
    J --> K[Relu]
    J --> L[Sigmoid]
    J --> M[Tanh]
    J --> N[SoftMax_CrossEntropy]
    C --> O[MSE]
    C --> P[CrossEntropy]
    C --> Q[CrossEntropy_SoftMax]
    R[Matrix] -->|contains| S[Matrix Operations]
    S --> T[dot]
    S --> U[transpose]
    S --> V[padding]
    S --> W[reshape]
    S --> X[get]
    S --> Y[set_matrix_1_to_x]
    S --> Z[sum]
    S --> AA[row_sum]
    S --> AB[row_max]
    S --> AC[exp_]
    S --> AD[log_]
    S --> AE[operator+]
    S --> AF[operator-]
    S --> AG[operator*]
    S --> AH[operator/]
```

## 主要類別

### MyFrame
主控制類別，負責訓練過程和管理層。

### Layer
基礎類別，所有層的基礎。

### ConvLayer, DropoutLayer, DenseLayer, FlattenLayer
具體層類別，實現不同的神經網絡層。

### LossFunc
損失函數的基礎類別，包含具體的損失函數如 MSE 和 CrossEntropy。

### ActiveFunc
激活函數的基礎類別，包含具體的激活函數如 Relu 和 Sigmoid。

### Matrix
用於數值計算的矩陣類別，包含各種矩陣操作。

## 使用方法

1. **編譯和運行**:
   - 使用支持 C++ 的編譯器編譯 `main.cpp` 和 `Matrix.h`。
   - 確保所有依賴的庫都已正確安裝。

2. **訓練模型**:
   - 修改 `main.cpp` 中的參數以設置模型的架構和訓練參數。
   - 運行程式以開始訓練過程。
