# 位置编码实现，基于旋转位置编码（RoPE）
import torch
# 预先计算频率和相位，供后续使用
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    # 1. 计算频率：1 / (theta^(2i/dim))，维度为 dim//2，因为每两个维度共享一个频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 2. 生成位置序列 t = [0, 1, ..., end-1]，维度为 (end,)
    t = torch.arange(end, device=freqs.device)
    # 3. 计算相位：t 和 freqs 的外积，得到一个形状为 (end, dim//2) 的矩阵
    freqs = torch.outer(t, freqs).float()
    # 4. 转换为复数形式 (cos(theta) + i*sin(theta))，得到一个形状为 (end, dim//2) 的复数矩阵
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # 将 Q/K 向量视为复数
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # 准备广播
    freqs_q = reshape_for_broadcast(freqs_cis, xq_)  # 针对 Q 的广播视图
    
    # 复数乘法即为旋转
    xq_out = torch.view_as_real(xq_ * freqs_q).flatten(3)
    
    # K 向量可能与 Q 向量有不同的头数（GQA），所以需单独生成广播视图
    freqs_k = reshape_for_broadcast(freqs_cis, xk_)
    xk_out = torch.view_as_real(xk_ * freqs_k).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xq)

# code/C6/llama2/src/rope.py
if __name__ == "__main__":
    # 准备参数和输入
    batch_size, seq_len, n_heads, n_kv_heads, head_dim = 4, 16, 8, 2, 16
    dim = n_heads * head_dim
    n_rep = n_heads // n_kv_heads

    # --- Test precompute_freqs_cis ---
    print("--- Test precompute_freqs_cis ---")
    freqs_cis = precompute_freqs_cis(dim=head_dim, end=seq_len * 2)
    print("freqs_cis shape:", freqs_cis.shape)

    # --- Test apply_rotary_emb ---
    print("\n--- Test apply_rotary_emb ---")
    xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
    xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)
    freqs_cis_slice = freqs_cis[:seq_len]
    xq_out, xk_out = apply_rotary_emb(xq, xk, freqs_cis_slice)
    print("xq shape (in/out):", xq.shape, xq_out.shape)
    print("xk shape (in/out):", xk.shape, xk_out.shape)
