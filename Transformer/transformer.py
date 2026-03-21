# Transformer 核心模块实现，包含编码器层、解码器层和整体框架
import torch
import torch.nn as nn
import math
# 导入组件
from attention import MultiHeadAttention
from ffn import FeedForward
from norm import LayerNorm
from pos import PositionalEncoding

'''
class SelfAttention(nn.Module):
    """自注意力模块"""
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size                        # 定义Q、K、V的维度
        self.q_linear = nn.Linear(hidden_size, hidden_size)   # 定义线性层来生成Query、Key和Value向量
        self.k_linear = nn.Linear(hidden_size, hidden_size)
        self.v_linear = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        # 将k矩阵的最后两个维度进行转置，(seq_len, hidden_size) -> (hidden_size, seq_len)，以便与q进行矩阵乘法。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        attention_weights = torch.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v) # 将注意力权重与v矩阵相乘，得到加权的输出，即上下文向量。
        
        return context
'''

# 编码器层
class EncoderLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力子层
        self.attention = MultiHeadAttention(dim, n_heads, dropout)
        self.attention_norm = LayerNorm(dim)
        # 前馈网络子层
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.ffn_norm = LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 子层 1：自注意力
        _x = x
        x = self.attention(x, x, x, mask)  # Q=K=V=x
        x = self.attention_norm(_x + self.dropout(x))
        
        # 子层 2：前馈网络
        _x = x
        x = self.feed_forward(x)
        x = self.ffn_norm(_x + self.dropout(x))
        
        return x
    
# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, dropout=0.1):
        super().__init__()
        # 1. 带掩码的自注意力
        self.self_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.self_attn_norm = LayerNorm(dim)
        # 2. 交叉注意力
        self.cross_attention = MultiHeadAttention(dim, n_heads, dropout)
        self.cross_attn_norm = LayerNorm(dim)
        # 3. 前馈网络
        self.feed_forward = FeedForward(dim, hidden_dim, dropout)
        self.ffn_norm = LayerNorm(dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        # 子层 1：带掩码的自注意力
        _x = x
        x = self.self_attention(x, x, x, tgt_mask)
        x = self.self_attn_norm(_x + self.dropout(x))
        
        # 子层 2：交叉注意力（Q 来自解码器，K/V 来自编码器输出）
        _x = x
        x = self.cross_attention(x, enc_output, enc_output, src_mask)
        x = self.cross_attn_norm(_x + self.dropout(x))
        
        # 子层 3：前馈网络
        _x = x
        x = self.feed_forward(x)
        x = self.ffn_norm(_x + self.dropout(x))
        
        return x


# transformer整体框架
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 dim=512,  n_heads=8, n_layers=6, 
                 hidden_dim=2048, max_seq_len=5000, dropout=0.1):
        super().__init__()
    
        self.dim = dim
        # 1. 嵌入层与位置编码
        # src_embedding: 将源语言序列映射为向量 (Encoder输入)
        self.src_embedding = nn.Embedding(src_vocab_size, dim)
        # tgt_embedding: 将目标语言序列映射为向量 (Decoder输入)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim)
        self.pos_encoder = PositionalEncoding(dim, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # 2. 编码器与解码器堆叠
        # 使用 ModuleList 来存储层列表，支持按索引访问和自动注册参数
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        ])
        
        # 3. 输出头
        self.output = nn.Linear(dim, tgt_vocab_size)

        self._init_parameters()
    
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        # src_mask: [batch, 1, 1, src_len]，pad token 假设为 0
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: [batch, 1, tgt_len, tgt_len]，结合 pad mask 和 causal mask
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
        tgt_subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        tgt_mask = tgt_pad_mask & tgt_subsequent_mask.unsqueeze(0)
        return src_mask, tgt_mask

    def encode(self, src, src_mask):
        x = self.src_embedding(src) * math.sqrt(self.dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        x = self.tgt_embedding(tgt) * math.sqrt(self.dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        # 1. 生成掩码 (Padding Mask & Causal Mask)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 2. 编码器前向传播
        enc_output = self.encode(src, src_mask)
        
        # 3. 解码器前向传播
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # 4. 输出 Logits
        return self.output(dec_output)
        return logits

