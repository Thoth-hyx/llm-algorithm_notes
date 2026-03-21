import torch
import os
# 导入 Hugging Face 的 AutoTokenizer 和 AutoModel 类，用于加载预训练的 BERT 模型和对应的分词器
from transformers import AutoTokenizer, AutoModel
# 1. 环境和模型配置
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

device = torch.device("cpu")
model_name = "bert-base-chinese"
texts = ["我来自中国", "我喜欢自然语言处理"]

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name) # 加载预训练的 BERT 分词器，负责将文本转换为模型输入所需的 token ids 和 attention masks
model = AutoModel.from_pretrained(model_name).to(device) # 加载预训练的 BERT 模型，并将其移动到指定的设备（GPU 或 CPU）上进行计算
model.eval()

print("\n--- BERT 模型结构 ---")
print(model)

# 3. 文本预处理
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

# 打印 Tokenizer 的完整输出，以理解其内部结构
print("\n--- Tokenizer 输出 ---")
for key, value in inputs.items():
    print(f"{key}: \n{value}\n")

# 4. 模型推理
with torch.no_grad():
    outputs = model(**inputs) # 将预处理后的输入传递给 BERT 模型进行前向传播，得到模型的输出结果，通常包括 last_hidden_state 和 pooler_output 等信息

# 5. 提取特征
last_hidden_state = outputs.last_hidden_state   # last_hidden_state 是 BERT 模型的输出之一，包含了每个输入 token 的隐藏状态（特征向量），其形状通常为 (batch_size, sequence_length, hidden_size)
sentence_features_pooler = getattr(outputs, "pooler_output", None) # pooler_output 是 BERT 模型的另一个输出，通常是对 [CLS] token 的特征向量进行池化后的结果，形状为 (batch_size, hidden_size)

# (1) 提取句子级别的特征向量 ([CLS] token)
sentence_features = last_hidden_state[:, 0, :]

# (2) 提取第一个句子的词元级别特征
first_sentence_tokens = last_hidden_state[0, 1:6, :]


print("\n--- 特征提取结果 ---")
print(f"句子特征 shape: {sentence_features.shape}")
if sentence_features_pooler is not None:
    print(f"pooler_output shape: {sentence_features_pooler.shape}")
print(f"第一个句子的词元特征 shape: {first_sentence_tokens.shape}")
