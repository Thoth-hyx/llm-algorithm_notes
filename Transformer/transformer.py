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
        ''' dim: 模型维度 (hidden size)，n_heads: 注意力头数，hidden_dim: 前馈网络隐藏层维度，dropout: dropout率 '''
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
        _x = x                             # 保存输入以便残差连接
        x = self.attention(x, x, x, mask)  # Q=K=V=x
        x = self.attention_norm(_x + self.dropout(x))
        
        # 子层 2：前馈网络
        _x = x
        x = self.feed_forward(x)
        x = self.ffn_norm(_x + self.dropout(x))
        
        return x # x 表示编码器层的输出，经过自注意力和前馈网络处理后的结果，实际上是一个新的表示，包含了输入序列的上下文信息。
    
# 解码器层
class DecoderLayer(nn.Module):
    def __init__(self, dim, n_heads, hidden_dim, dropout=0.1):
        ''' dim: 模型维度 (hidden size)，n_heads: 注意力头数，hidden_dim: 前馈网络隐藏层维度，dropout: dropout率 '''
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
        ''' x: 解码器输入 (目标序列的嵌入)，enc_output: 编码器输出，src_mask: 编码器输入掩码，tgt_mask: 解码器输入掩码'''
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
        ''' src_vocab_size: 源语言词表大小，tgt_vocab_size: 目标语言词表大小，
            dim: 模型维度 (hidden size)，n_heads: 注意力头数，n_layers: 编码器/解码器层数，
            hidden_dim: 前馈网络隐藏层维度，max_seq_len: 最大序列长度，dropout: dropout率 '''
        super().__init__() # 调用父类(nn.Module)的初始化方法
    
        self.dim = dim
        # 1. 嵌入层与位置编码
        # src_embedding: 将源语言序列映射为向量 (Encoder输入)
        self.src_embedding = nn.Embedding(src_vocab_size, dim)

        # tgt_embedding: 将目标语言序列映射为向量 (Decoder输入)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim)

        # pos_encoder: 位置编码模块，给输入序列添加位置信息
        self.pos_encoder = PositionalEncoding(dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)
        
        # 2. 编码器与解码器堆叠
        # 使用 ModuleList 来存储层列表，支持按索引访问和自动注册参数
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(dim, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        ]) # 这里创建了 n_layers 个编码器层，每个层都是 EncoderLayer 的实例，参数相同但权重不同。
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(dim, n_heads, hidden_dim, dropout) for _ in range(n_layers)
        ]) # 同样创建 n_layers 个解码器层，每个层都是 DecoderLayer 的实例，参数相同但权重不同。
        
        # 3. 输出头，将解码器输出映射到目标词表大小的维度，得到每个位置的词概率分布
        self.output = nn.Linear(dim, tgt_vocab_size)

        self._init_parameters()
    
    def _init_parameters(self):
        '''使用 Xavier 初始化方法来初始化模型参数,适用于线性层和注意力层的权重,避免训练初期出现梯度消失'''
        for p in self.parameters():   # 递归遍历当前模型和所有子模块（编码器层、解码器层、多头注意力层、线性层、嵌入层等）的所有 nn.Parameter 类型可训练参数
            if p.dim() > 1:           # 只初始化维度 > 1 的参数（矩阵），跳过 1 维参数（偏置）
                nn.init.xavier_uniform_(p)

    def generate_mask(self, src, tgt):
        ''' src: [batch, src_len]，tgt: [batch, tgt_len] '''
        # src_mask: [batch, 1, 1, src_len]，pad token 假设为 0
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # tgt_mask: [batch, 1, tgt_len, tgt_len]，结合 pad mask 和 causal mask
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, tgt_len]
        # 生成下三角矩阵，确保解码器只能看到当前位置及之前的位置，防止信息泄露
        tgt_subsequent_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()  # [tgt_len, tgt_len]
        tgt_mask = tgt_pad_mask & tgt_subsequent_mask.unsqueeze(0) # [batch, 1, tgt_len, tgt_len]
        return src_mask, tgt_mask

    def encode(self, src, src_mask):
        ''' src: [batch, src_len]，src_mask: [batch, 1, 1, src_len] '''
        x = self.src_embedding(src) * math.sqrt(self.dim)  # 将输入序列转换为嵌入向量，并进行缩放 (乘以 sqrt(dim))，以保持数值稳定性
        x = self.pos_encoder(x)                            # 添加位置编码，给嵌入向量添加位置信息，使模型能够区分不同位置的词
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, src_mask)   # 依次通过每个编码器层，传入当前层的输入和源序列掩码，得到编码器的输出表示
        return x

    def decode(self, tgt, enc_output, src_mask, tgt_mask):
        ''' tgt: [batch, tgt_len]，enc_output: [batch, src_len, dim]，src_mask: [batch, 1, 1, src_len]，tgt_mask: [batch, 1, tgt_len, tgt_len] '''
        x = self.tgt_embedding(tgt) * math.sqrt(self.dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

    def forward(self, src, tgt):
        ''' src: [batch, src_len]，tgt: [batch, tgt_len] '''
        # 1. 生成掩码 (Padding Mask & Causal Mask)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 2. 编码器前向传播
        enc_output = self.encode(src, src_mask)
        
        # 3. 解码器前向传播
        dec_output = self.decode(tgt, enc_output, src_mask, tgt_mask)
        
        # 4. 输出 Logits
        return self.output(dec_output)
        return logits

