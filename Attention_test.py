# 基于循环神经网络的Seq2Seq模型，包含两种注意力机制：无参数和带参数的注意力模块。
import torch.nn as nn
import torch
import numpy as np
import random

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 使用双向LSTM
        )
        #  双向LSTM输出的维度是 hidden_size * 2，需要通过线性层降维到 hidden_size，以便与解码器匹配
        self.fc = nn.Linear(hidden_size * 2, hidden_size) 

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # 将双向RNN的输出通过线性层降维，使其与解码器维度匹配
        outputs = torch.tanh(self.fc(outputs)) 

        return outputs, hidden, cell # 返回所有时间步的输出和最终的隐藏状态、细胞状态
    

class AttentionSimple(nn.Module):
    """1: 无参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionSimple, self).__init__()
        # 注册一个缓冲区张量，确保缩放因子是一个不可训练的参数，并且在模型保存和加载时正确处理
        self.register_buffer("scale_factor", torch.sqrt(torch.FloatTensor([hidden_size])))
    
    # 利用相似度越高的向量，其点积越大，来计算注意力权重
    def forward(self, hidden, encoder_outputs):
        # hidden shape: (num_layers, batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, src_len, hidden_size)
        
        # Q: 解码器最后一层的隐藏状态
        query = hidden[-1].unsqueeze(1)  # -> (batch, 1, hidden)
        # K/V: 编码器的所有输出
        keys = encoder_outputs  # -> (batch, src_len, hidden)

        # energy shape: (batch, 1, src_len) 直接计算 query 和 keys 的点积，energy反映了 query 与每个 encoder_outputs 的相似度。
        energy = torch.bmm(query, keys.transpose(1, 2)) / self.scale_factor
        # transpose(1, 2) 将 keys 的维度从 (batch, src_len, hidden) 转换为 (batch, hidden, src_len)，以便与 query 进行（bmm）批量矩阵乘法。
        # attention_weights shape: (batch, src_len)
        return torch.softmax(energy, dim=2).squeeze(1) # 返回最终的注意力权重
    
class AttentionParams(nn.Module):
    """2: 带参数的注意力模块"""
    # 创建一个小型神经网络来学习Query和Keys之间的关系，允许模型学习更复杂的注意力模式，而不仅仅是基于点积的相似度。
    def __init__(self, hidden_size):
        super(AttentionParams, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1] # 获取输入序列的长度
        hidden_last_layer = hidden[-1].unsqueeze(1).repeat(1, src_len, 1) # 将解码器最后一层的隐藏状态复制 src_len 次，使其能与每一个编码器状态进行配对
        # energy shape: (batch, src_len, hidden)，通过一个线性层和非线性激活函数来计算每个时间步的能量值，反映了 query 与每个 encoder_outputs 的关系。
        energy = torch.tanh(self.attn(torch.cat((hidden_last_layer, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2) # 表示 query 与每个 encoder_outputs 的相关性。
        
        return torch.softmax(attention, dim=1)

# 通用解码器
class DecoderWithAttention(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, attention_module):
        super(DecoderWithAttention, self).__init__()
        self.attention = attention_module
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size * 2,  # 输入维度是 词嵌入(hidden_size) + 上下文向量(hidden_size)
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden, cell, encoder_outputs):
        embedded = self.embedding(x.unsqueeze(1))

        # 1. 计算注意力权重
        # a shape: [batch,1,src_len] attention可以是AttentionSimple或AttentionParams的实例，计算得到的a是一个权重分布，表示解码器当前状态与编码器每个时间步输出的相关性。
        a = self.attention(hidden, encoder_outputs).unsqueeze(1)
        
        # 2. 计算上下文向量
        context = torch.bmm(a, encoder_outputs)

        # 3. 将上下文向量与当前输入拼接
        rnn_input = torch.cat((embedded, context), dim=2)

        # 4. 传入RNN解码
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        
        # 5. 预测输出
        predictions = self.fc(outputs.squeeze(1))
        
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    """带注意力的Seq2Seq"""
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        # 适配Encoder(双向)和Decoder(单向)的状态维度
        hidden = hidden.view(self.encoder.rnn.num_layers, 2, batch_size, -1).sum(dim=1)
        cell = cell.view(self.encoder.rnn.num_layers, 2, batch_size, -1).sum(dim=1)

        input = trg[:, 0]
        for t in range(1, trg_len):
            # 在循环的每一步，都将 encoder_outputs 传递给解码器
            # 这是 Attention 机制能够"回顾"整个输入序列的关键
            output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[:, t] if teacher_force else top1
            
        return outputs

# 验证函数，检查编码器输出、注意力权重计算、Seq2Seq前向传播和Loss计算的正确性。
def verify_attention_seq2seq():
    # 1. 设定超参数
    vocab_size = 10    # 词表大小
    hidden_size = 16   # 隐藏状态维度
    num_layers = 1     # LSTM层数
    batch_size = 2     # 批次大小
    src_len = 5        # 输入序列长度
    trg_len = 6        # 输出序列长度   
    device = torch.device('cpu')

    # 2. 实例化模型
    encoder = Encoder(vocab_size, hidden_size, num_layers)
    
    # 分别测试两种注意力机制
    attn_simple = AttentionSimple(hidden_size)
    attn_params = AttentionParams(hidden_size)
    
    # 选择 AttentionSimple 注意力机制进行测试
    decoder_simple = DecoderWithAttention(vocab_size, hidden_size, num_layers, attn_simple)
    model = Seq2Seq(encoder, decoder_simple, device)

    # 3. 构造伪数据
    src = torch.randint(0, vocab_size, (batch_size, src_len))
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))

    # 4. 验证编码器输出
    print("--- 验证编码器 ---")
    enc_out, h, c = encoder(src)
    print(f"Encoder Outputs shape: {enc_out.shape}") # (B, src_len, hidden)
    assert enc_out.shape == (batch_size, src_len, hidden_size)

    # 5. 验证注意力权重计算
    print("\n--- 验证注意力权重 ---")
    # 模拟一个解码器的隐状态
    dec_hidden = torch.randn(num_layers, batch_size, hidden_size)
    
    # 测试 AttentionSimple
    weights_simple = attn_simple(dec_hidden, enc_out)
    print(f"Simple Attention weights shape: {weights_simple.shape}") # (B, src_len)
    assert torch.allclose(weights_simple.sum(dim=1), torch.ones(batch_size), atol=1e-5), "AttentionSimple 权重未归一化！"
    
    # 测试 AttentionParams
    weights_params = attn_params(dec_hidden, enc_out)
    print(f"Params Attention weights shape: {weights_params.shape}")
    assert torch.allclose(weights_params.sum(dim=1), torch.ones(batch_size), atol=1e-5), "AttentionParams 权重未归一化！"
    print("注意力权重归一化检查通过！")

    # 6. 验证 Seq2Seq 整体前向传播
    print("\n--- 验证 Seq2Seq 前向传播 ---")
    outputs = model(src, trg)
    print(f"Seq2Seq Output shape: {outputs.shape}") # (B, trg_len, vocab_size)
    assert outputs.shape == (batch_size, trg_len, vocab_size)
    
    # 7. 验证 Loss 计算
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs[:, 1:, :].reshape(-1, vocab_size), trg[:, 1:].reshape(-1))
    print(f"模拟 Loss: {loss.item():.4f}")
    
    print("\n恭喜！所有逻辑验证通过，模型结构搭建正确。")

# 执行验证
verify_attention_seq2seq()