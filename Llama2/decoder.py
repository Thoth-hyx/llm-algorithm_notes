import torch
import torch.nn as nn
from config import ModelConfig
from gqa import Attention
from mlp import MLP
from norm import RMSNorm
from gqa import precompute_freqs_cis

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig):
        super().__init__()
        # 定义多头注意力的头数
        self.n_heads = args.n_heads
        # 定义输入维度
        self.dim = args.dim
        # 定义每个头的维度，等于输入维度除以头数
        self.head_dim = args.dim // args.n_heads
        # 定义LLaMA2Attention对象，用于进行多头注意力计算
        self.attention = Attention(args)
        # 定义LLaMAMLP对象，用于进行前馈神经网络计算
        self.feed_forward = MLP(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        # 定义层的ID
        self.layer_id = layer_id
        # 定义注意力计算的归一化层
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        # 定义前馈神经网络计算的归一化层
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        # 前向传播函数
        # 首先，输入x经过注意力归一化层，然后进行注意力计算，结果与输入x相加得到h
        # 然后，h经过前馈神经网络归一化层，然后进行前馈神经网络计算，结果与h相加得到输出
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


if __name__ == "__main__":
    args = ModelConfig()
    
    # 创建LLaMADecoderLayer实例
    decoderlayer = DecoderLayer(0, args)

    # 模拟输入数据
    dim = args.dim
    seq_len = 50

    x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

    freqs_cis = precompute_freqs_cis(dim//args.n_heads, seq_len)

    out = decoderlayer(x, freqs_cis)

    print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]
