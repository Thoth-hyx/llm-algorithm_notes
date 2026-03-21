import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # 定义 Wq, Wk, Wv 矩阵
        # 这里我们使用一个大的线性层一次性计算所有头的 Q, K, V
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        
        # 最终输出的线性层 Wo
        self.wo = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        # 1. 线性投影
        # [batch, seq_len, dim] -> [batch, seq_len, dim]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # 2. 分头 (Split Heads)
        # view 操作将q、k、v从(batch_size, seq_len, hidden_size)转换为(batch_size, seq_len, num_heads, head_dim)
        # 然后转置: [batch, n_heads, seq_len, head_dim] 以便并行计算
        q = q.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 3. 计算缩放点积注意力 (Scaled Dot-Product Attention)
        # scores: [batch, n_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 4. 应用掩码 (Masking)
        if mask is not None:
            # mask == 0 的位置被填充为负无穷，Softmax 后变为 0
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 5. Softmax 与加权求和
        attn_weights = torch.softmax(scores, dim=-1)
        
        if self.dropout is not None:
             attn_weights = self.dropout(attn_weights)
             
        # context: [batch, n_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # 6. 合并多头 (Concat Heads)
        # [batch, n_heads, seq_len, head_dim] -> [batch, seq_len, n_heads, head_dim]
        # -> [batch, seq_len, dim]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        
        # 7. 输出层投影
        output = self.wo(context)
        
        return output

if __name__ == "__main__":
    # 准备参数
    batch_size, seq_len, dim = 2, 10, 512
    n_heads = 8
    
    # 初始化模块
    mha = MultiHeadAttention(dim, n_heads)
    
    # 准备输入 (Query, Key, Value 相同)
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = mha(x, x, x)
    
    # 验证输出
    print("--- MultiHeadAttention Test ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")