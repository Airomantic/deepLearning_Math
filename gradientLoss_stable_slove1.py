import numpy as np
import matplotlib.pyplot as plt

# 设置初始参数
np.random.seed(42)
m = 100  # 样本数
n = 10   # 特征数
iterations = 100
learning_rate = 0.1

# 生成数据
X = np.random.randn(m, n) # A function that generates a matrix of shape (m, n), where each elements is randomly drawn from the standard normal distribution (mean=0, standard deviation = 1)
w_star = np.random.randn(n, 1)
y = X @ w_star + 0.1 * np.random.randn(m, 1)

# 定义线性化神经网络的损失函数和梯度
def linearized_loss(w, X, y, w_star):
    predictions = X @ w_star + X @ (w - w_star)
    return np.mean((y - predictions)**2)

def gradient(w, X, y, w_star):
    predictions = X @ w_star + X @ (w - w_star)
    return -2 * X.T @ (y - predictions) / m

# 初始化权重
w_gd = np.random.randn(n, 1)
w_sgd = np.copy(w_gd)

# 记录损失
loss_gd = []
loss_sgd = []

# 梯度下降和随机梯度下降
for i in range(iterations):
    # 计算梯度
    grad_gd = gradient(w_gd, X, y, w_star)
    epsilon = np.random.randn(n, 1) * np.sqrt(1 / m)
    
    # 更新权重
    w_gd -= learning_rate * grad_gd
    w_sgd -= learning_rate * (grad_gd + epsilon)
    
    # 记录损失
    loss_gd.append(linearized_loss(w_gd, X, y, w_star))
    loss_sgd.append(linearized_loss(w_sgd, X, y, w_star))

# 作图
plt.figure(figsize=(14, 7))
plt.plot(loss_gd, label='Gradient Descent')
plt.plot(loss_sgd, label='Stochastic Gradient Descent')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss vs Iterations for GD and SGD')
plt.legend()
plt.grid(True)
plt.savefig("picture/gradient_loss.png")
plt.show()
