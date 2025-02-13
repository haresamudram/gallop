from typing import List, Optional, Type

import torch
from torch import Tensor

NoneType = Type[None]


def topk_reduce(
    local_logits: Tensor,
    topk: Optional[List[int]] = None,
) -> Tensor:
    """
    local_logits is a Tensor of shape (b, p, k, 1) or (b, p, k, m)
    """
    if topk is None:
        return local_logits

    _, n_patches, _, n_prompts = local_logits.shape
    
    assert len(topk) == n_prompts or len(topk) == 1, "Please provide a k for each local prompt or one for all."

    maxk = min(max(topk), n_patches)

    local_logits = local_logits.topk(dim=1, k=maxk)[0]

    # Local prompt 1 focus on the top 5 patches while local prompt 2 focus on the next 10 patches.
    custom_loss = True

    if len(topk) == 1:
        local_logits = local_logits.mean(dim=1)
    else:
        if custom_loss:
            prompt1 = local_logits[:, :topk[0], :, 0].mean(dim=1)
<<<<<<< Updated upstream
            prompt2 = local_logits[:, topk[0]:topk[1], :, 1].mean(dim=1)
            local_logits = torch.stack([prompt1, prompt2], dim=-1)
=======
            #prompt2 = local_logits[:, topk[0]:topk[1], :, 1].mean(dim=1)
            #local_logits = torch.stack([prompt1, prompt2], dim=-1)
            #local_logits = (local_logits[...,0] + local_logits[...,1]) / 2
            local_logits = prompt1
>>>>>>> Stashed changes
        else :
            local_logits = torch.stack([local_logits[:, :k, :, i].mean(dim=1) for i, k in enumerate(topk)], dim=-1)

    return local_logits
