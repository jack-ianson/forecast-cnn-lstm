import torch


def stack_months(months: list[torch.Tensor]) -> torch.Tensor:
    """
    Stacks multiple monthly tensors along the first dimension.

    Args:
        *months (list[torch.Tensor]): List of monthly tensors to stack.

    Returns:
        torch.Tensor: A tensor containing all months stacked along the first dimension.
    """
    return torch.stack(months, dim=0)
