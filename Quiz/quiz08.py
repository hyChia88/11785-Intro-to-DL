# import numpy as np
# import matplotlib.pyplot as plt

# # 定义循环权重矩阵
# W_r = np.array([
#     [0.75, 0.25, 0],
#     [0.2, -0.1, 0.7],
#     [-0.2, 0.65, 0.15]
# ])

# # 定义输入向量 (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
# input_vector = np.ones(3) / np.sqrt(3)

# # 定义tanh激活函数
# def tanh_activation(x):
#     return np.tanh(x)

# # 计算向量的范数（长度）
# def vector_norm(v):
#     return np.linalg.norm(v)

# # 模拟RNN
# def simulate_rnn(W_r, input_vector, num_steps=50):
#     # 初始隐藏状态为零向量
#     h_prev = np.zeros(3)
    
#     # 存储隐藏状态和范数
#     hidden_states = []
#     norms = []
    
#     # 第一个时间步：处理输入
#     # 由于W_x是单位矩阵且激活函数是tanh
#     h = tanh_activation(input_vector)
    
#     hidden_states.append(h.copy())
#     norms.append(vector_norm(h))
    
#     print(f"t=0: h={h}, norm={norms[0]:.6f}")
    
#     # 后续时间步：无额外输入
#     for t in range(1, num_steps + 1):
#         # z_t = W_r * h_{t-1}
#         z = np.dot(W_r, h)
        
#         # h_t = tanh(z_t)
#         h = tanh_activation(z)
        
#         hidden_states.append(h.copy())
#         norms.append(vector_norm(h))
        
#         print(f"t={t}: h={h}, norm={norms[t]:.6f}")
        
#         # 检查范数是否已经降至初始值的1/100以下
#         if norms[t] < norms[0] / 100:
#             print(f"记忆持续时间: {t} 时间步")
#             return hidden_states, norms, t
    
#     print(f"记忆持续时间超过 {num_steps} 时间步")
#     return hidden_states, norms, None

# # 计算理论记忆持续时间（基于特征值）
# def calculate_theoretical_duration(W_r):
#     # 计算特征值
#     eigenvalues = np.linalg.eigvals(W_r)
#     print("特征值:", eigenvalues)
    
#     # 特征值的绝对值
#     abs_eigenvalues = np.abs(eigenvalues)
#     print("特征值绝对值:", abs_eigenvalues)
    
#     # 找到主导特征值（最大绝对值）
#     dominant_eigenvalue = np.max(abs_eigenvalues)
#     print(f"主导特征值: {dominant_eigenvalue:.6f}")
    
#     # 如果主导特征值<1，计算理论记忆持续时间
#     if dominant_eigenvalue < 1:
#         theoretical_duration = np.ceil(np.log(0.01) / np.log(dominant_eigenvalue))
#         print(f"基于主导特征值的理论记忆持续时间: {theoretical_duration:.0f} 时间步")
#         return theoretical_duration
#     else:
#         print("主导特征值 >= 1，网络不会遗忘输入")
#         return float('inf')

# # 运行模拟和理论计算
# print("特征值分析:")
# theoretical_duration = calculate_theoretical_duration(W_r)

# print("\n模拟RNN:")
# hidden_states, norms, memory_duration = simulate_rnn(W_r, input_vector)

# # 可视化结果
# # plt.figure(figsize=(12, 6))

# # # 绘制隐藏状态范数随时间变化
# # plt.subplot(1, 2, 1)
# # plt.plot(norms, marker='o')
# # plt.axhline(y=norms[0]/100, color='r', linestyle='--', label='阈值 (1/100)')
# # plt.title('隐藏状态范数随时间变化')
# # plt.xlabel('时间步')
# # plt.ylabel('范数')
# # plt.grid(True)
# # plt.legend()

# # # 绘制范数比率（相对于初始值）
# # plt.subplot(1, 2, 2)
# # ratios = [norm/norms[0] for norm in norms]
# # plt.semilogy(ratios, marker='o')  # 使用对数尺度
# # plt.axhline(y=0.01, color='r', linestyle='--', label='阈值 (1/100)')
# # plt.title('隐藏状态范数比率（对数尺度）')
# # plt.xlabel('时间步')
# # plt.ylabel('比率（相对于初始值）')
# # plt.grid(True)
# # plt.legend()

# # plt.tight_layout()
# # plt.show()

# # 总结结果
# print("\n结果总结:")
# print(f"初始隐藏状态范数: {norms[0]:.6f}")
# print(f"阈值 (1/100 初始值): {norms[0]/100:.6f}")
# print(f"记忆持续时间: {memory_duration} 时间步")
# print(f"理论记忆持续时间: {theoretical_duration:.0f} 时间步")

import numpy as np
import matplotlib.pyplot as plt

# 定义循环权重矩阵
W_r = np.array([
    [0.75, 0.25, 0],
    [0.2, -0.1, 0.7],
    [-0.2, 0.65, 0.15]
])

# 定义输入向量 (1/sqrt(3), 1/sqrt(3), 1/sqrt(3))
input_vector = np.ones(3) / np.sqrt(3)

# 定义ReLU激活函数（替换原来的tanh）
def relu_activation(x):
    return np.maximum(0, x)

# 计算向量的范数（长度）
def vector_norm(v):
    return np.linalg.norm(v)

# 模拟RNN
def simulate_rnn(W_r, input_vector, num_steps=50):
    # 初始隐藏状态为零向量
    h_prev = np.zeros(3)
    
    # 存储隐藏状态和范数
    hidden_states = []
    norms = []
    
    # 第一个时间步：处理输入
    # 由于W_x是单位矩阵且激活函数是ReLU
    h = relu_activation(input_vector)
    
    hidden_states.append(h.copy())
    norms.append(vector_norm(h))
    
    print(f"t=0: h={h}, norm={norms[0]:.6f}")
    
    # 后续时间步：无额外输入
    for t in range(1, num_steps + 1):
        # z_t = W_r * h_{t-1}
        z = np.dot(W_r, h)
        
        # h_t = relu(z_t)
        h = relu_activation(z)
        
        hidden_states.append(h.copy())
        norms.append(vector_norm(h))
        
        print(f"t={t}: h={h}, norm={norms[t]:.6f}")
        
        # 检查范数是否已经降至初始值的1/100以下
        if norms[t] < norms[0] / 100:
            print(f"记忆持续时间: {t} 时间步")
            return hidden_states, norms, t
    
    print(f"记忆持续时间超过 {num_steps} 时间步")
    return hidden_states, norms, None

# 分析使用ReLU时的网络行为
def analyze_relu_behavior(W_r, input_vector, num_steps=50):
    """分析使用ReLU时的特殊情况"""
    h = relu_activation(input_vector)
    print(f"初始隐藏状态: {h}")
    
    # 检查是否会出现所有激活值为0的情况
    zero_activations = False
    
    for t in range(1, 6):  # 只看前几步
        z = np.dot(W_r, h)
        print(f"时间步 {t}, 加权和 z = {z}")
        
        h = relu_activation(z)
        print(f"时间步 {t}, 隐藏状态 h = {h}")
        
        if np.all(h == 0):
            print(f"在时间步 {t} 所有激活值变为0！")
            zero_activations = True
            break
    
    # 对于ReLU网络的特殊考虑
    if zero_activations:
        print("\n注意: 使用ReLU时，如果所有隐藏单元的输入变为负值，")
        print("      所有激活将变为0，且此后网络将保持在0状态。")
        print("      在这种情况下，记忆持续时间为激活首次全部变为0的时间步。")
    
    # 检查特征值和ReLU网络的关系
    eigenvalues = np.linalg.eigvals(W_r)
    print("\n权重矩阵特征值:", eigenvalues)
    print("特征值绝对值:", np.abs(eigenvalues))
    
    print("\n注意: 与tanh不同，ReLU网络的动态行为不仅由特征值决定，")
    print("      还取决于输入和权重的符号模式，因为ReLU会将负值归零。")

# 运行模拟
print("分析ReLU激活的特殊行为:")
analyze_relu_behavior(W_r, input_vector)

print("\n模拟使用ReLU的RNN:")
hidden_states, norms, memory_duration = simulate_rnn(W_r, input_vector, 100)

# 可视化结果
plt.figure(figsize=(12, 8))

# 绘制隐藏状态范数随时间变化
plt.subplot(2, 2, 1)
plt.plot(norms, marker='o')
plt.axhline(y=norms[0]/100, color='r', linestyle='--', label='阈值 (1/100)')
plt.title('隐藏状态范数随时间变化 (ReLU)')
plt.xlabel('时间步')
plt.ylabel('范数')
plt.grid(True)
plt.legend()

# 绘制范数比率（相对于初始值）
plt.subplot(2, 2, 2)
ratios = [norm/norms[0] for norm in norms]
plt.semilogy(ratios, marker='o')  # 使用对数尺度
plt.axhline(y=0.01, color='r', linestyle='--', label='阈值 (1/100)')
plt.title('隐藏状态范数比率（对数尺度）')
plt.xlabel('时间步')
plt.ylabel('比率（相对于初始值）')
plt.grid(True)
plt.legend()

# 绘制各个隐藏单元的激活值
plt.subplot(2, 1, 2)
hidden_states_array = np.array(hidden_states)
for i in range(3):
    plt.plot(hidden_states_array[:, i], label=f'隐藏单元 {i+1}')
plt.title('各隐藏单元激活值随时间变化')
plt.xlabel('时间步')
plt.ylabel('激活值')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 总结结果
print("\n结果总结:")
print(f"初始隐藏状态范数: {norms[0]:.6f}")
print(f"阈值 (1/100 初始值): {norms[0]/100:.6f}")

if memory_duration:
    print(f"记忆持续时间: {memory_duration} 时间步")
else:
    print(f"记忆持续时间超过模拟步数或模式不同于tanh激活")
    
    # 对于ReLU网络的进一步分析
    if np.any([np.all(h == 0) for h in hidden_states]):
        zero_state_time = next(i for i, h in enumerate(hidden_states) if np.all(h == 0))
        print(f"在时间步 {zero_state_time} 所有激活值变为0")
        print("对于ReLU网络，一旦所有激活值变为0，将永远保持在0状态")
        print(f"因此记忆持续时间为 {zero_state_time} 时间步")