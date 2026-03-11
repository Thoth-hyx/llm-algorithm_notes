import torch
import torch.nn as nn
import numpy as np

# 编码器
class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers):
        super(Encoder, self).__init__()
        # 定义词嵌入层和LSTM层
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, # 词汇表大小
            embedding_dim=hidden_size  # 词嵌入维度与LSTM的输入维度设置相同
        )
        self.rnn = nn.LSTM(
            input_size=hidden_size,    # LSTM的输入维度等于词嵌入维度
            hidden_size=hidden_size,   # LSTM的隐藏状态维度
            num_layers=num_layers,     # LSTM层数
            batch_first=True,          # 批次维度在前
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
        # 将 LSTM 输出的 hidden_size 维度的隐藏状态，映射到 vocab_size 维度的向量上，全连接层内部是一个[vocab_size, hidden_size]的权重矩阵
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, hidden, cell):
        # x shape: (batch_size)，只包含当前时间步的token
        x = x.unsqueeze(1) # -> (batch_size, 1)

        embedded = self.embedding(x)
        # 第一步(hidden, cell)是由Encoder传入的初始状态，之后每一步都由上一步Decoder传入
        # outputs shape: (batch_size, 1, hidden_size)，cell数据不对外输出，只在内部传递，hidden shape: (num_layers, batch_size, hidden_size)，这里hidden中hidden_size和outputs的hidden_size是一样的
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))  
        predictions = self.fc(outputs.squeeze(1)) # 数据维度：(batch_size, vocab_size)
        return predictions, hidden, cell

# Seq2Seq包装模块,实现教师强制训练模式
class Seq2Seq(nn.Module):
    '''Seq2Seq训练模式。'''
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
        # 创建一个形状为 (批次大小, 目标句子长度, 词表大小) 的全零张量，用于存储解码器在每一个时间步的输出。
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        # 调用编码器处理源序列 src，得到初始的上下文向量
        hidden, cell = self.encoder(src)

        # 启动解码，第一个输入是 <SOS>
        input = trg[:, 0]

        # 循环解码
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input, hidden, cell) # output shape: (batch_size, vocab_size)
            outputs[:, t, :] = output

            # 决定是否使用 Teacher Forcing
            teacher_force = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            # 如果 teacher_force，下一个输入是真实值；否则是模型的预测值
            input = trg[:, t] if teacher_force else top1

        return outputs
    
    def greedy_decode(self, src, max_len=12, sos_idx=1, eos_idx=2):
        """推理模式下的高效贪心解码。"""
        # 关闭训练模式，切断反向传播
        self.eval()
        with torch.no_grad():
            hidden, cell = self.encoder(src) # 获取初始上下文
            trg_indexes = [sos_idx]
            for _ in range(max_len):
                # 1. 输入只有上一个时刻的词元
                trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(self.device) #  PyTorch 需要的 Tensor 格式
                
                # 2. 解码一步，并传入上一步的状态
                output, hidden, cell = self.decoder(trg_tensor, hidden, cell)
                
                # 3. 获取当前步的预测，并更新状态用于下一步
                pred_token = output.argmax(1).item()  # .item() 将单元素张量转换为 Python 标量
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
        context = hidden_ctx[-1].unsqueeze(1).repeat(1, embedded.shape[1], 1) # 数据维度：(batch_size, 1, hidden_size)
        rnn_input = torch.cat((embedded, context), dim=2) # 数据维度：(batch_size, 1, hidden_size + hidden_size) 把 embedded 和 context 在最后一个维度上拼接

        # 解码器的初始状态 hidden, cell 在第一步可设为零；之后需传递并更新上一步状态
        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs.squeeze(1))
        return predictions, hidden, cell

# -------------------------------验证代码逻辑是否正确-------------------------------
def verify_logic():
    # 1. 设定超参数
    vocab_size = 10   # 设置一个小词表
    hidden_size = 8
    num_layers = 1
    batch_size = 2    # 批次为2
    src_len = 5       # 输入序列长度
    trg_len = 6       # 输出序列长度（包括 <SOS> 和 <EOS>）
    device = torch.device('cpu')

    # 2. 实例化模型
    encoder = Encoder(vocab_size, hidden_size, num_layers)
    decoder = Decoder(vocab_size, hidden_size, num_layers)
    model = Seq2Seq(encoder, decoder, device)

    # 3. 构造伪数据 
    src = torch.randint(0, vocab_size, (batch_size, src_len)) # 数据维度：(batch_size, src_len)，范围在词表大小内
    trg = torch.randint(0, vocab_size, (batch_size, trg_len))

    # 4. 测试 Seq2Seq 的前向传播
    print("正在测试 Seq2Seq 前向传播...")
    outputs = model(src, trg, teacher_forcing_ratio=0.5)
    print(f"输出结果形状: {outputs.shape}") # 应该是 (batch_size, trg_len, vocab_size)
    assert outputs.shape == (batch_size, trg_len, vocab_size), "Seq2Seq 输出维度错误！"
    print("Seq2Seq 逻辑验证通过！")

    # 5. 测试 DecoderAlt (上下文拼接版本)
    print("\n正在测试 DecoderAlt...")
    decoder_alt = DecoderAlt(vocab_size, hidden_size, num_layers)
    
    # 构造一次步长为 1 的解码输入
    x = torch.randint(0, vocab_size, (batch_size,))
    hidden_ctx = torch.randn(num_layers, batch_size, hidden_size) # 模拟 Encoder 的 hidden
    h_init = torch.zeros(num_layers, batch_size, hidden_size)
    c_init = torch.zeros(num_layers, batch_size, hidden_size)
    
    pred, h_new, c_new = decoder_alt(x, hidden_ctx, h_init, c_init)
    print(f"DecoderAlt 输出形状: {pred.shape}") # 应该是 (batch_size, vocab_size)
    assert pred.shape == (batch_size, vocab_size), "DecoderAlt 输出维度错误！"
    print("DecoderAlt 逻辑验证通过！")

    # 6. 测试简单的损失计算 (逻辑验证)
    criterion = nn.CrossEntropyLoss()
    # 展平以便计算损失: (B*T, V) -> (B*T)
    # 跳过第一个位置 <SOS>
    loss = criterion(outputs[:, 1:, :].reshape(-1, vocab_size), trg[:, 1:].reshape(-1))
    print(f"模拟计算 Loss: {loss.item():.4f}")
    print("\n恭喜，代码逻辑验证完全通过！")

# 执行验证
verify_logic()

    

