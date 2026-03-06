import numpy as np
import torch
import torch.nn as nn
# (B, T, E, H) 分别表示 批次/序列长度/输入维度/隐藏维度
B,E,H = 1,128,3

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

# 基于numpy实现RNN
def manual_rnn_numpy(x_np,U_np,W_np):
    B_local, T_local, _ = x_np.shape
    # 初始化 h_0 为零向量
    h_prev = np.zeros((B_local, H), dtype=np.float32) # 隐藏层维度（1，3）
    steps = []
    # 按时间步循环
    for t in range(T_local):
        x_t = x_np[:, t, :] # x_t为行向量
        # 核心公式实现
        h_t = np.tanh(x_t @ U_np + h_prev @ W_np)
        steps.append(h_t)
        h_prev = h_t # 更新状态
        
    return np.stack(steps, axis=1), h_prev

# pytorch中的RNN实现
def pytorch_rnn_forward(x, U, W):
    rnn = nn.RNN(
        input_size=E,        # 词嵌入的维度
        hidden_size=H,       # 隐藏层的节点数
        num_layers=1,        # RNN 的层数
        nonlinearity='tanh', # 激活函数
        bias=False,          # 是否使用偏置项
        batch_first=True,    # 维度顺序参数,默认为false:[T,B,E],true:[B,T,E]
        bidirectional=False, # 是否构建一个双向RNN
    )
    with torch.no_grad():
        # PyTorch 内部存放的是转置后的权重
        rnn.weight_ih_l0.copy_(U.T)
        rnn.weight_hh_l0.copy_(W.T)
    y, h_n = rnn(x)          # y:输出序列,h_n:最终的隐藏状态
    return y, h_n.squeeze(0)

# 将NumPy结果转回PyTorch张量
out_manual = torch.from_numpy(out_manual_np)

# 使用 allclose 进行浮点数精度下的严格比较
print("逐步输出一致:", torch.allclose(out_manual, out_torch, atol=1e-6))
# 输出: True