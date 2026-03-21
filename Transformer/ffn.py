# 标准的 Transformer FFN 是一个简单的两层全连接网络，中间包含激活函数
import torch.nn as nn
import torch

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)  # 升维
        self.w2 = nn.Linear(hidden_dim, dim)  # 降维
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # 线性变换 -> ReLU -> Dropout -> 线性变换
        return self.w2(self.dropout(torch.relu(self.w1(x))))

if __name__ == "__main__":
    # 准备参数
    batch_size, seq_len, dim = 2, 10, 512
    hidden_dim = 2048
    
    # 初始化模块
    ffn = FeedForward(dim, hidden_dim)
    
    # 准备输入
    x = torch.randn(batch_size, seq_len, dim)
    
    # 前向传播
    output = ffn(x)
    
    # 验证输出
    print("--- FeedForward Test ---")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
