from typing import List, Optional, Type

import torch
from torch import Tensor

NoneType = Type[None]

# Weighting function
def frequency_weight(frequency):
    return 0.4 + torch.sigmoid(frequency - 1)

def topk_reduce_frequency(
    local_logits: Tensor,
    topk: Optional[List[int]] = None,
) -> Tensor:
    """
    local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
    """
    if topk is None:
        return local_logits
    batch_size, num_classes, num_prompt= local_logits.shape[0], 1000, local_logits.shape[3]
    final_logits = torch.zeros(batch_size, num_classes, device=local_logits.device)  # (b x 1000)
    max_logits, pred_classes = local_logits.max(dim=2)  # max_logits: (b x 196 x 4), pred_classes: (b x 196 x 4)
    
    for prompt_idx, top_k in enumerate(topk):
        # Extract max logits and class IDs for the current prompt
        prompt_logits = max_logits[:, :, prompt_idx]  # (B x 196)
        prompt_classes = pred_classes[:, :, prompt_idx]  # (B x 196)

        # Find top-k patches for each batch
        top_k_indices = torch.topk(prompt_logits, k=top_k, dim=1).indices  # (B x top_k)

        # Compute weights for each batch
        prompt_weighted_logits = torch.zeros(batch_size, num_classes, device=local_logits.device)  # (B x 1000)
        for b in range(batch_size):
            # Extract top-k class IDs and logits for the batch
            batch_top_k_classes = prompt_classes[b, top_k_indices[b]]  # (top_k,)
            batch_top_k_logits = local_logits[b, top_k_indices[b], :, prompt_idx]  # (top_k x 1000)

            # Compute frequency of each class in top-k
            unique_classes, counts = torch.unique(batch_top_k_classes, return_counts=True)

            # Compute weights
            weights = torch.zeros(top_k, device=local_logits.device)  # (top_k,)
            for class_id, count in zip(unique_classes, counts):
                weights[batch_top_k_classes == class_id] = frequency_weight(count.float())

            # Apply weights and compute mean logits for top-k patches
            mean_logits = batch_top_k_logits.mean(dim=0)  # (1000,)
            prompt_weighted_logits[b] = mean_logits * weights.mean()  # Weighted logits for the batch

        # Aggregate weighted logits for the current prompt
        final_logits += prompt_weighted_logits

    # Average across all prompts
    final_logits /= num_prompts  # Shape: (B x 1000)
    return final_logits