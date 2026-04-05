from config import ModelConfig
from rope import apply_rotary_emb, precompute_freqs_cis, reshape_for_broadcast, repeat_kv    
import torch
import torch.nn as nn
import math
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, args: ModelConfig):
        super().__init__()  # 调用父类的构造函数，初始化 nn.Module 的内部状态
        # 根据是否指定n_kv_heads，确定用于键（key）和值（value）的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        # 确保总头数可以被键值头数整除。
        assert args.n_heads % self.n_kv_heads == 0

        # 模型并行处理大小，默认为1，即不进行模型并行处理。
        model_parallel_size = 1
        # 本地计算头数，等于总头数除以模型并行处理大小。
        self.n_local_heads = args.n_heads // model_parallel_size
        # 本地键值头数，等于键值头数除以模型并行处理大小。
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        # 重复次数，用于扩展键和值的尺寸。
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        # 每个头的维度，等于模型维度除以头的总数。
        self.head_dim = args.dim // args.n_heads

        # 定义权重矩阵。
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)     # 查询权重矩阵，矩阵维度为 (dim, n_heads * head_dim)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)   # 键权重矩阵，矩阵维度为 (dim, n_kv_heads * head_dim)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)   # 值权重矩阵，矩阵维度为 (dim, n_kv_heads * head_dim)
        # 输出权重矩阵。
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)   # 输出权重矩阵，矩阵维度为 (n_heads * head_dim, dim)

        # 定义dropout。
        self.attn_dropout = nn.Dropout(args.dropout)    # 注意力权重的dropout，通常在计算注意力权重后应用，以增加模型的正则化能力。
        self.resid_dropout = nn.Dropout(args.dropout)   # 残差连接的dropout，通常在输出后应用，以增加模型的正则化能力。
        # 保存dropout概率。
        self.dropout = args.dropout

        # 检查是否使用Flash Attention（需要PyTorch >= 2.0）。
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            # 若不支持Flash Attention，则使用手动实现的注意力机制，并设置mask。
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 创建一个上三角矩阵，用于遮蔽未来信息。
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = x.shape

        # 计算查询（Q）、键（K）、值（V）。
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        # 调整形状以适应头的维度。
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)      
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # 应用旋转位置嵌入（RoPE）。
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        # 对键和值进行扩展以适应重复次数。
        xk = repeat_kv(xk, self.n_rep)
        xv = repeat_kv(xv, self.n_rep)

        # 将头作为批次维度处理。
        xq = xq.transpose(1, 2)  # 转置为 (batch_size, n_local_heads, seq_len, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 根据是否支持Flash Attention，选择实现方式。
        if self.flash:
            # 使用Flash Attention。参数包括查询、键、值，以及是否使用dropout和是否为因果注意力。
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # 使用手动实现的注意力机制。
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)  # 转置回 (batch_size, seq_len, n_heads * head_dim)，并合并头维度

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


if __name__ == "__main__":
    args = ModelConfig()
    # 创建Attention实例
    attention_model = Attention(args)

    # 模拟输入数据
    batch_size = 1
    seq_len = 50  # 假设实际使用的序列长度为50
    dim = args.dim
    x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
    # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
    # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

    freqs_cis = precompute_freqs_cis(dim//args.n_heads, seq_len)

    # 运行Attention模型
    output = attention_model(x, freqs_cis)

    # attention出来之后的形状 依然是[batch_size, seq_len, dim]
    print("Output shape:", output.shape)
