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
    
    # Dimensions
    # b: Number of rows in the input tensor `local_logits`
    # local_logits: Input tensor of shape [b, 128, 1000, 2]
    # prompt1: Tensor extracted from the last dimension of `local_logits`, shape [b, 128, 1000]
    # prompt2: Tensor extracted from the last dimension of `local_logits`, shape [b, 128, 1000]
    # local_logits_prompt1: Result tensor for the first 5 columns, shape [128, 1000]
    # local_logits_prompt2: Result tensor for the remaining valid columns, shape [128, 1000]

    b = local_logits.shape[0]

    # Initialize tensors to accumulate results
    local_logits_prompt1 = torch.zeros((b, 1000), device='cuda')  # For first 5 columns
    local_logits_prompt2 = torch.zeros((b, 1000), device='cuda')  # For remaining valid columns

    # Extract the first tensor (prompt1) from local_logits and compute top-5 indices
    prompt1 = local_logits[..., 0]  # Shape: [b, 128, 1000]
    max_values = torch.max(prompt1, dim=2)[0]  # Shape: [b, 128]
    indices_prompt_1 = max_values.topk(dim=-1, k=topk[0])[1]  # Top-5 indices, shape: [b, 5]

    # Extract the second tensor (prompt2) from local_logits and compute top-30 indices
    prompt2 = local_logits[..., 1]  # Shape: [b, 128, 1000]
    max_values = torch.max(prompt2, dim=2)[0]  # Shape: [b, 128]
    indices_prompt_2 = max_values.topk(dim=-1, k=topk[1])[1]  # Top-30 indices, shape: [b, 30]

    # Each row in the batch
    for row in range(b):
        count = 0  # Count of valid columns for local_logits_prompt2

        for col in range(30):
            # Check if the column from indices_prompt_2 is not in indices_prompt_1
            #if indices_prompt_2[row, col] not in indices_prompt_1[row, :]:
            #    # Accumulate corresponding values from prompt2
            #    local_logits_prompt2[row, :] += prompt2[row, indices_prompt_2[row, col].tolist(), :]
            #    count += 1  # Increment valid column count

            # Average the top 30
            local_logits_prompt2[row, :] += prompt2[row, indices_prompt_2[row, col].tolist(), :]
            
            # Accumulate the first 5 columns for prompt1
            if col < 5:
                local_logits_prompt1[row, :] += prompt1[row, indices_prompt_1[row, col].tolist(), :]

    # Normalize results
    local_logits_prompt1 /= 5  # Average over the first 5 columns
    local_logits_prompt2 /= 30  # Average over the valid columns
    local_logits = local_logits_prompt1 * local_logits_prompt2
    return local_logits
    