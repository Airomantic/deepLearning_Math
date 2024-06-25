import numpy as np

# 参数设置
b = 32          # 小批量大小
delta = 0.1     # 给定的 delta 值
s = 0.01        # 学习率

# 计算不等式右侧的值
right_hand_side = np.sqrt(b / delta / s)

# 打印结果
print(f"Right hand side of the inequality: {right_hand_side}")

# 计算随机生成的 Sigma 矩阵的 Frobenius 范数（这里用随机生成的数据代替实际数据）
n = 10  # 假设 Sigma 是一个 n x n 的矩阵
Sigma = np.random.randn(n, n)
Sigma_F_norm = np.linalg.norm(Sigma, 'fro')

# 比较左右两侧的值
if Sigma_F_norm <= right_hand_side:
    print("The inequality is satisfied.")
else:
    print("The inequality is not satisfied.")
