
Question 2：
![](picture/Prove_gradientDescent_locallyStable.jpg)
对于github：
1. $公式在有逗号后面的的情况下不能被渲染
2. “双”$符合后面跟斜杆的不能被渲染，“单”$可以
3. 注意w^\star 不是六角的*
# Question 2：Deep neural network gradient descent method and random gradient descent method analysis

## Problem overview
# 问题2：深度神经网络的梯度下降法和随机梯度下降法分析

hypothesis $F(x; w)$ is a deep neural network with output scalars，where $x$ is the input and $ w $ represents the weight. Let $F$ be continuously differentiable with respect to $w$ , and for training data $(x_j, y_j)_{j=1}^m$ is over-parameterized : There exists $w^\star$ such that for all j, $F(x_j, w^\star) = y_i$. In order to study the local optimization dynamics at w* when training neural networks, we consider linearized neural networks 
$$ 
\tilde{F}(x; w) 
$$
, its the Loss function $\text{Loss}(w)$:

$$
\text{Loss}(w) := \frac{1}{2m} \sum_{j=1}^m (y_j - \tilde{F}(x_j; w))^2
$$

where $\hat{y}_j(w) = F(x_j; w)$。

Define the gradient descent update rule:

$$
w_{i+1} = w_i - s \nabla \text{Loss}(w_i)
$$

And random gradient descent update rules:

$$
w_{i+1} = w_i - s (\nabla \text{Loss}(w_i) + \epsilon_i)
$$

Where $\epsilon_i$ is the noise term, assuming 
$E\epsilon_i = 0$
and $E\epsilon_i \epsilon_i^T = M(w_i)/b$, where 
$b$ is the size of the mini-batch.

Suppose the method matrix $M$ is:

$$
\Sigma = \frac{1}{m} \sum_{j=1}^m \nabla F(x_j, w^\star ) \nabla F(x_j,  w^\star )^T
$$

Aligned in the following sense:

$$
\frac{\text{Tr}(M(w) \Sigma)}{2 \text{Loss}(w) \| \Sigma \|_F^2} \geq \delta
$$

True for $\delta > 0$ and all $w$. Here $\| \cdot \|_F$ represents the Frobenius norm.

## 题目要求

1. For gradient descent, it is proved that if the spectral norm of ∑ is satisfied:

$$
\| \Sigma \|_2 \leq \frac{2}{s}
$$

So gradient descent is locally stable (i.e. $\text{Loss}(w) $ is bounded for all $i $). 注意这里的i.e 是对于所有的i

2. For the stochastic gradient descent method, assuming that $\text{Loss}(w) $ is bounded for all $i $, the following inequality must hold:

$$
\| \Sigma \|_F \leq \sqrt{\frac{b/ \delta }{s}}
$$
