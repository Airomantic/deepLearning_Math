
Question 2：
![](picture/Prove_gradientDescent_locallyStable.jpg)

# Question 2：Deep neural network gradient descent method and random gradient descent method analysis

## Problem overview

假设 \( F(x; w) \) 是一个输出标量的深度神经网络，其中 \( x \) 是输入，\( w \) 表示权重。设 \( F \) 关于 \( w \) 连续可微，并且对于训练数据 \( (x_j, y_j)_{j=1}^m \) 进行最小化损失函数 \( \text{Loss}(w) \):

\[ \text{Loss}(w) := \frac{1}{2m} \sum_{j=1}^m (y_j - \hat{y}_j(w))^2 \]

其中 \( \hat{y}_j(w) = F(x_j; w) \)。

定义梯度下降法更新规则：

\[ w_{i+1} = w_i - s \nabla \text{Loss}(w_i) \]

以及随机梯度下降法更新规则：

\[ w_{i+1} = w_i - s (\nabla \text{Loss}(w_i) + \epsilon_i) \]

其中，\(\epsilon_i\) 是噪声项，假设 \( E[\epsilon_i] = 0 \) 和 \( E[\epsilon_i \epsilon_i^T] = M(w_i) / b \)，这里 \( b \) 是 mini-batch 的大小。

假设方法矩阵 \( M \) 为：

\[ \Sigma = \frac{1}{m} \sum_{j=1}^m \nabla F(x_j, w^*) \nabla F(x_j, w^*)^T \]

在以下意义上对齐：

\[ \frac{\text{Tr}(M(w) \Sigma)}{2 \text{Loss}(w) \| \Sigma \|_F^2} \geq \delta \]

对于 \( \delta > 0 \) 和所有 \( w \) 成立。这里 \( \| \cdot \|_F \) 表示 Frobenius 范数。

## 题目要求

1. 对于梯度下降法，证明如果 \( 2\delta \) 适当选择，则：

\[ \| \epsilon_i \|_2 \leq \frac{s}{2} \]

那么梯度下降是局部稳定的（即对所有 \( i \)，\(\text{Loss}(w_i)\) 是有界的）。

2. 对于随机梯度下降法，假设 \( \text{Loss}(w_i) \) 对所有 \( i \) 有界，则以下不等式必须成立：

\[ \| \epsilon_i \|_F \leq \sqrt{\frac{b}{s}} \]
