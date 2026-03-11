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
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, (hidden, cell) = self.rnn(embedded)
        
        # 将双向RNN的输出通过线性层降维，使其与解码器维度匹配
        outputs = torch.tanh(self.fc(outputs))

        return outputs, hidden, cell
    

class AttentionSimple(nn.Module):
    """1: 无参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionSimple, self).__init__()
        # 确保缩放因子是一个 non-learnable buffer
        self.register_buffer("scale_factor", torch.sqrt(torch.FloatTensor([hidden_size])))

    def forward(self, hidden, encoder_outputs):
        # hidden shape: (num_layers, batch_size, hidden_size)
        # encoder_outputs shape: (batch_size, src_len, hidden_size)
        
        # Q: 解码器最后一层的隐藏状态
        query = hidden[-1].unsqueeze(1)  # -> (batch, 1, hidden)
        # K/V: 编码器的所有输出
        keys = encoder_outputs  # -> (batch, src_len, hidden)

        # energy shape: (batch, 1, src_len)
        energy = torch.bmm(query, keys.transpose(1, 2)) / self.scale_factor
        
        # attention_weights shape: (batch, src_len)
        return torch.softmax(energy, dim=2).squeeze(1)
    
class AttentionParams(nn.Module):
    """2: 带参数的注意力模块"""
    def __init__(self, hidden_size):
        super(AttentionParams, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden_last_layer = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((hidden_last_layer, encoder_outputs), dim=2)))
        attention = torch.sum(self.v * energy, dim=2)
        
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
        # a shape: [batch, src_len]
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


