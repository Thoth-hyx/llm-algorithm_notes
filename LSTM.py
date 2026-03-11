import numpy as np
import torch
import torch.nn as nn
# (B, T, E, H) 分别表示 批次/序列长度/输入维度/隐藏维度
B,E,H = 1,128,3

# 手动实现 sigmoid 函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 输入准备，例子：播放周杰伦的《稻香》
def prepare_inputs():
    np.random.seed(42)                                   # 每个词生成一个随机的词向量
    vocab = {"播放": 0, "周杰伦": 1, "的": 2, "《稻香》": 3}
    tokens = ["播放", "周杰伦", "的", "《稻香》"]
    ids = [vocab[t] for t in tokens]

    # 词向量表: (V, E)
    V = len(vocab)
    emb_table = np.random.randn(V, E).astype(np.float32) # 数据维度（4，128）

    # 取出序列词向量并加上 batch 维度: (B, T, E)
    x_np = emb_table[ids][None]                          # 数据维度（1，4，128）
    return tokens, x_np

# 手动实现 LSTM 
def manual_lstm_numpy(x_np, weights):
    U_i, W_i, U_f, W_f, U_c, W_c, U_o, W_o = weights   # 遗忘门，输入门，候选记忆，输出门权重
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)  # 隐藏状态，短期记忆
    c_prev = np.zeros((B_local, H), dtype=np.float32)  # 细胞状态，长期记忆

    steps = []
    # 按时间步循环
    for t in range(T_local):
        x_t = x_np[:, t, :]  # x_t为行向量
        # 1. 遗忘门
        f_t = sigmoid(x_t @ U_f + h_prev @ W_f)  # 遗忘门使用sigmoid激活函数

        # 2. 输入门与候选记忆
        i_t = sigmoid(x_t @ U_i + h_prev @ W_i)  # 输入门使用sigmoid激活函数
        c_tilde_t = np.tanh(x_t @ U_c + h_prev @ W_c)  # 候选记忆使用tanh激活函数

        # 3. 更新细胞状态
        c_t = f_t * c_prev + i_t * c_tilde_t           # 细胞状态更新公式
        
        # 4. 输出门与隐藏状态
        o_t = sigmoid(x_t @ U_o + h_prev @ W_o)  # 输出门使用sigmoid激活函数
        h_t = o_t * np.tanh(c_t)                       # 隐藏状态更新公式
        
        steps.append(h_t)
        h_prev, c_prev = h_t, c_t
        
    outputs = np.stack(steps, axis=1)
    return outputs, h_prev, c_prev

# 1. 准备数据
tokens, x_np = prepare_inputs()
x_torch = torch.from_numpy(x_np)

# 2. 定义 PyTorch LSTM
lstm_torch = nn.LSTM(
    input_size=E, 
    hidden_size=H, 
    num_layers=1, 
    bias=False, 
    batch_first=True,
    )

# 3. 提取 PyTorch 随机生成的权重
# PyTorch 的权重存储格式是 (4*H, E) 和 (4*H, H)，顺序是 IFCO (Input, Forget, Cell, Output)
with torch.no_grad():
    # 获取权重 Tensor
    weight_ih = lstm_torch.weight_ih_l0 # (12, 128) -> 4个3x128
    weight_hh = lstm_torch.weight_hh_l0 # (12, 3)   -> 4个3x3

    # split(H) 会把大矩阵切分成 4 块，每块大小为 H
    # 顺序：i (input), f (forget), c (cell/gate), o (output)
    U_chunk = weight_ih.split(H) 
    W_chunk = weight_hh.split(H)
    
    # 转换为 NumPy 并转置 (.T)
    # PyTorch 内部是 x @ W.T，手动写的是 x @ W，所以这里要转置
    U_i = U_chunk[0].numpy().T
    U_f = U_chunk[1].numpy().T
    U_c = U_chunk[2].numpy().T
    U_o = U_chunk[3].numpy().T
    
    W_i = W_chunk[0].numpy().T
    W_f = W_chunk[1].numpy().T
    W_c = W_chunk[2].numpy().T
    W_o = W_chunk[3].numpy().T

# 打包权重传给 Numpy 函数
weights_np = (U_i, W_i, U_f, W_f, U_c, W_c, U_o, W_o)

# 4. 运行 NumPy 版本
out_manual_np, h_last_np, c_last_np = manual_lstm_numpy(x_np, weights_np)
out_manual = torch.from_numpy(out_manual_np)
h_last_manual = torch.from_numpy(h_last_np)
c_last_manual = torch.from_numpy(c_last_np)

# 5. 运行 PyTorch 版本
out_torch, (h_n_torch, c_n_torch) = lstm_torch(x_torch)

# 6. 验证结果
print(f"NumPy 输出形状: {out_manual.shape}")
print(f"PyTorch 输出形状: {out_torch.shape}")

# 比较所有时间步的输出
match_out = torch.allclose(out_manual, out_torch, atol=1e-6)
# 比较最后的 h
match_h = torch.allclose(h_last_manual, h_n_torch.squeeze(0), atol=1e-6)
# 比较最后的 c
match_c = torch.allclose(c_last_manual, c_n_torch.squeeze(0), atol=1e-6)

print("-" * 30)
print(f"序列输出(Output)一致: {match_out}")
print(f"最终隐状态(h_n)一致:   {match_h}")
print(f"最终细胞状态(c_n)一致: {match_c}")
print("-" * 30)

if not match_out:
    diff = (out_manual - out_torch).abs().max()
    print(f"最大误差: {diff.item()}")
else:
    print("匹配！手动 LSTM 逻辑正确。")