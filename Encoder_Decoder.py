import torch
import torch.nn as nn
import numpy as np

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        # 定义词嵌入层和LSTM层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)
        # 返回最终的隐藏状态和细胞状态作为上下文
        _, (hidden, cell) = self.rnn(embedded)  # _是每一个时间步的输出，不需要
        return hidden, cell

# 解码器
class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        # 将 LSTM 输出的 hidden_size 维度的隐藏状态，映射到 vocab_size 维度的向量上
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size)，只包含当前时间步的token
        x = x.unsqueeze(1) # -> (batch_size, 1)

        embedded = self.embedding(x)
        # 接收上一步的状态 (hidden, cell)，计算当前步
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        predictions = self.fc(outputs.squeeze(1)) # -> (batch_size, vocab_size)
        return predictions, hidden, cell

# Seq2Seq包装模块,实现教师强制训练模式
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    # 源序列src，目标序列trg
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.fc.out_features
        # 创建一个形状为 (batch_size, trg_len, vocab_size) 的全零张量，用于存储解码器在每一个时间步的输出。
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        # 调用编码器处理源序列 src，得到初始的上下文向量
        hidden, cell = self.encoder(src)

        # 启动解码，第一个输入是 <SOS>
        input = trg[:, 0]

        # 循环解码
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell)
            outputs[:, t, :] = output

            # 决定是否使用 Teacher Forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # 如果 teacher_force，下一个输入是真实值；否则是模型的预测值
            input = trg[:, t] if teacher_force else top1

        return outputs
    
    def greedy_decode(self, src, max_len=12, sos_idx=1, eos_idx=2):
        """推理模式下的高效贪心解码。"""
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src) # 获取初始上下文
            trg_indexes = [sos_idx]
            for _ in range(max_len):
                # 1. 输入只有上一个时刻的词元
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device)
                
                # 2. 解码一步，并传入上一步的状态
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
                
                # 3. 获取当前步的预测，并更新状态用于下一步
                pred_token = output.argmax(1).item()
                trg_indexes.append(pred_token)
                if pred_token == eos_idx:
                    break
        return trg_indexes
    
# 改进版本的解码器，将上下文向量作为解码器每个时间步的额外输入
class DecoderAlt(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(DecoderAlt, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size
        )
        # 主要改动 1: RNN的输入维度是 词嵌入+上下文向量
        self.rnn = nn.LSTM(
            input_size=hidden_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden_ctx, hidden, cell):
        x = x.unsqueeze(1)
        embedded = self.embedding(x)

        # 主要改动 2: 将上下文向量与当前输入拼接
        # 这里简单地取编码器最后一层的 hidden state 作为上下文代表
        context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1)
        rnn_input = torch.cat((embedded, context), dim=2)

        # 解码器的初始状态 hidden, cell 在第一步可设为零；之后需传递并更新上一步状态
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell

    

