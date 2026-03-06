import numpy as np
import torch
import torch.nn as nn
# (B, T, E, H) 分别表示 批次/序列长度/输入维度/隐藏维度
B,E,H = 1,128,3

def manual_lstm_numpy(x_np, weights):
    U_f, W_f, U_i, W_i, U_c, W_c, U_o, W_o = weights   # 遗忘门，输入门，候选记忆，输出门权重
    B_local, T_local, _ = x_np.shape
    h_prev = np.zeros((B_local, H), dtype=np.float32)  # 隐藏状态，短期记忆
    c_prev = np.zeros((B_local, H), dtype=np.float32)  # 细胞状态，长期记忆

    steps = []
    # 按时间步循环
    for t in range(T_local):
        x_t = x_np[:, t, :]
        # 1. 遗忘门
        f_t = sigmoid(x_t @ U_f + h_prev @ W_f)

        # 2. 输入门与候选记忆
        i_t = sigmoid(x_t @ U_i + h_prev @ W_i)
        c_tilde_t = np.tanh(x_t @ U_c + h_prev @ W_c)

        # 3. 更新细胞状态
        c_t = f_t * c_prev + i_t * c_tilde_t
        
        # 4. 输出门与隐藏状态
        o_t = sigmoid(x_t @ U_o + h_prev @ W_o)
        h_t = o_t * np.tanh(c_t)
        
        steps.append(h_t)
        h_prev, c_prev = h_t, c_t
        
    outputs = np.stack(steps, axis=1)
    return outputs, h_prev, c_prev