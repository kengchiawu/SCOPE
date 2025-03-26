import math
import torch
import warnings
from typing import List, Optional, Tuple, Union
from transformers.utils import is_flash_attn_2_available
from einops import rearrange
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

def score_cover_topk(x: torch.Tensor, score: float):
    cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
    topk = torch.sum(cumsum_x <= score, dim=-1) + 1
    # torch.save(x,f"/homeB/youkangqi/SCOPE/results/llama-3.1-8b-instruct_2048_eager/attn_score/head_wise/layer_0_prelen_2598.pt")
    # raise ValueError(f"{topk.shape}")
    return topk

def get_headwise_budget(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        block_size: int = 1,
        chunk_size: int = 16,
        prefill_max_budget: int = 2048,
        prefill_min_budget: int = 128,
        gamma: float = 0.95,
        gqa_interleave: bool = False,
):
    batch_size, num_heads, q_len, head_dim = query.shape
    batch_size, num_heads, kv_len, head_dim = key.shape
    num_share_q_heads = num_heads // key.shape[2]

    last_q = query[:,  :,-block_size:, :]
    qk = torch.matmul(last_q,key.transpose(2,3))/math.sqrt(head_dim)
    '''flexprefill的点积运用不熟练
    if not gqa_interleave:
        qk = torch.einsum(
            "bhid, bhjd -> bhij",
            last_q#.view(last_q.shape[0], last_q.shape[1], -1, num_share_q_heads, head_dim)
            ,
            key#.view(key.shape[0], key.shape[1], -1, 1, head_dim),
        )
        # raise ValueError(f"\nqk:{qk.shape}\nkey:{key.shape}\nlast_q:{last_q.shape}") # qk:torch.Size([1, kv_len, 1, 1, num_heads])
    else:
        qk = torch.einsum(
            "bihgd, bjhgd -> bhgij",
            last_q.view(
                last_q.shape[0], last_q.shape[1], num_share_q_heads, -1, head_dim
            ),
            key.view(key.shape[0], key.shape[1], 1, -1, head_dim),
        )
    '''
    
    '''block_size暂时默认为1,不考虑causal_mask
    global causal_mask
    if causal_mask is None:
        causal_mask = torch.arange(0, block_size, device=last_q.device)
        causal_mask = causal_mask[:, None] >= causal_mask[None, :]
        causal_mask = causal_mask[None, None, None, ...]
    qk[..., -block_size:].masked_fill_(
        ~causal_mask[..., :block_size, :block_size], float("-inf")
    )
    '''
    # softmax，上采样到fp32
    qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
    # qk = rearrange(qk, "b h g i j -> b (h g) i j")
    # qk.shape = [batch_size, num_heads, block_size, key_len]
    num_tokens = score_cover_topk(qk,gamma)
    # raise ValueError(f"\nnum_tokens:{num_tokens.shape}\nqk:{qk.shape}")
    # 接下来是生成head_wise_budget，并思考如何进行head_wise_spare的分配
    # 初步想法是生成一个mask矩阵，对注意力分数排序后的indicates矩阵进行掩码
    # 该矩阵和head_wise_budget一样，生成一次就行
    # num_tokens.shape = [batch, num_heads, block_size]
    head_wise_budget = num_tokens.squeeze(dim=-1)# [batch, num_heads]
    # 计算归一化后的预算
    #head_wise_budget = head_wise_budget / head_wise_budget.max()

    # 缩放并向上取整
    #head_wise_budget = torch.ceil(head_wise_budget * prefill_max_budget)

    # 确保不低于最小预算
    head_wise_budget = torch.clamp(head_wise_budget, min=prefill_min_budget)
    # head_wise_budget[head_wise_budget<prefill_min_budget] = prefill_min_budget
    attn_score_3d = qk.mean(dim=-2) # [batch, num_heads, kv_len]
    
    # 对每个头的分数进行降序排序，获取排序后的索引
    sorted_indices = torch.argsort(attn_score_3d, dim=-1, descending=True)  # [batch, num_heads, kv_len]
    
    # 生成位置索引并与预算比较，生成前k的布尔掩码
    arange = torch.arange(kv_len, device=qk.device).view(1, 1, -1).expand(batch_size, num_heads, -1)
    budget_expanded = head_wise_budget.unsqueeze(-1)  # [batch, num_heads, 1]
    mask = arange < budget_expanded  # [batch, num_heads, kv_len]

    # 将排序后的掩码映射回原始位置
    final_mask = torch.zeros_like(attn_score_3d, dtype=torch.bool)
    final_mask.scatter_(-1, sorted_indices, mask)

    # 扩展掩码形状并应用到原始注意力分数
    final_mask = final_mask.unsqueeze(2) # [batch, num_heads, 1, kv_len]
    selected_scores = qk * final_mask
    # raise ValueError(f"\nfinal_mask:{final_mask.shape}")
    return head_wise_budget

class Head_Wise_Attention():
    prefill_length = 0
    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0
    
    def __init__(
            self,
            prefill_max_budget = 1024,
            prefill_min_budget = 128,
            decode_metric = 'None',
            decode_budget = 1024,
            chunk_size = 16,
            num_hidden_layers = 32,
            gamma = 0.95,
    ):
        self.prefill_max_budget = prefill_max_budget
        self.prefill_min_budget = prefill_min_budget
        self.decode_metric = decode_metric
        self.decode_budget = decode_budget
        self.chunk_size = chunk_size
        self.num_hidden_layers = num_hidden_layers
        self.gamma = gamma
        
    def reset(
            self,
            prefill_max_budget = 1024,
            decode_metric = 'None',
            decode_budget = 1024,
            chunk_size = 16,
            num_hidden_layers = 32,
    ):
        self.prefill_max_budget = prefill_max_budget
        self.decode_metric = decode_metric
        self.decode_budget = decode_budget
        self.chunk_size = chunk_size
        self.num_hidden_layers = num_hidden_layers

    def headwise_attention_computaion_prefill(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            # is_prefill: bool,
            softmax_scale: Optional[float] = None,
            return_computational_ratio: Optional[bool] = False,
    )-> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        batch_size, q_len, num_q_heads, head_dim = q.shape
        batch_size, kv_len, num_k_heads, head_dim = k.shape
        assert batch_size == 1, "only support batch size 1 for now"
        assert q_len > 1, "support for first token generation"

        head_wise_budget = get_headwise_budget(
            self,
            q,
            k,
            block_size = 1,
            chunk_size=self.chunk_size,
            prefill_max_budget=self.prefill_max_budget,
            prefill_min_budget=128,
            gamma=self.gamma,
        )

        return head_wise_budget # [batch_size, num_heads]

    def headwise_attention_computaion_decode(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            # is_prefill: bool,
            softmax_scale: Optional[float] = None,
            return_computational_ratio: Optional[bool] = False,
    )-> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
        q

def init_headwise_attention(self, num_hidden_layers):
    if not hasattr(self, "headwise_attention"):
        if not hasattr(self.config, 'decode_metric'):
            self.config.decode_metric = 'None'
        if not hasattr(self.config, 'prefill_max_budget'):
            self.config.prefill_max_budget = 2048
        if not hasattr(self.config, 'prefill_min_budget'):
            self.config.prefill_min_budget = 128
        if not hasattr(self.config, 'decode_budget'):
            self.config.decode_budget = 512
        if not hasattr(self.config, 'chunk_size'):
            self.config.chunk_size = 16
        if not hasattr(self.config, 'gamma'):
            self.config.gamma = 0.95
    
    self.headwise_attention = Head_Wise_Attention(
        prefill_max_budget = self.config.prefill_max_budget,
        prefill_min_budget = self.config.prefill_min_budget,
        decode_metric = self.config.decode_metric,
        decode_budget = self.config.decode_budget,
        chunk_size = self.config.chunk_size,
        num_hidden_layers = num_hidden_layers,
        gamma = self.config.gamma,
    )