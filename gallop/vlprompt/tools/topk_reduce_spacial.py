from typing import List, Optional, Type

import torch
from torch import Tensor

NoneType = Type[None]


def topk_reduce_spacial(
    local_logits: Tensor,
    topk: Optional[List[int]] = None,
) -> Tensor:
    """
    local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
    """
    if topk is None:
        return local_logits
    
    b = local_logits.shape[0]

    prompt_1 = local_logits[...,0]  # Shape: [b, 196, 1000]
    prompt_2 = local_logits[..., 1]  # Shape: [b, 196, 1000]
    
    max_logits_prompt_1 = torch.max(prompt_1, dim=2).values  # Shape: (128, 196)
    max_logits_prompt_2 = torch.max(prompt_2, dim=2).values  # Shape: (128, 196)
    
    # Find top 5 indices from prompt_1 and top 30 indices from prompt_2
    top5_indices = torch.topk(max_logits_prompt_1, k=topk[0], dim=1).indices  # Shape: (128, 5)
    top30_indices = torch.topk(max_logits_prompt_2, k=topk[1], dim=1).indices  # Shape: (128, 30)
    
    # Remove top 5 indices from top 30 indices
    mask = torch.ones_like(top30_indices, dtype=torch.bool)  # Shape: (128, 30)
    for batch_idx in range(b):
        mask[batch_idx] = ~torch.isin(top30_indices[batch_idx], top5_indices[batch_idx])

    filtered_indices_list = [top30_indices[i][mask[i]] for i in range(b)]
    max_valid_indices = 25
    filtered_indices = torch.zeros(top30_indices.size(0), max_valid_indices, dtype=torch.long, device='cuda')
    for i, indices in enumerate(filtered_indices): 
        filtered_indices[i, :len(indices)] = filtered_indices_list[i][:max_valid_indices]
    
    # Compute local_logit_prompt_1
    expanded_top5 = top5_indices.unsqueeze(-1).expand(-1, -1, prompt_1.size(2))  # Shape: (128, 5, 1000)
    top5_logits_prompt_1 = torch.gather(prompt_1, dim=1, index=expanded_top5)  # Shape: (128, 5, 1000)
    local_logits_prompt_1 = top5_logits_prompt_1.mean(dim=1)  # Shape: (128, 1000)
    
    # Compute local_logit_prompt_2
    expanded_filtered = filtered_indices.unsqueeze(-1).expand(-1, -1, prompt_2.size(2))  # Shape: (128, N, 1000)
    filtered_logits_prompt_2 = torch.gather(prompt_2, dim=1, index=expanded_filtered)  # Shape: (128, N, 1000)
    local_logits_prompt_2 = filtered_logits_prompt_2.mean(dim=1)
    
    local_logits = torch.stack([local_logits_prompt_1, local_logits_prompt_2], dim=-1)
    return local_logits
    