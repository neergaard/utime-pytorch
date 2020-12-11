import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, num_classes, smooth=1):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, pred, target, mask=None):

        assert (
            pred.shape == target.shape
        ), f"Target shape: {target.shape} does not match predicted shape: {pred.shape}!"

        if mask is None:
            mask = torch.full(pred.shape, True).type_as(pred)
        elif mask.shape[-1] != self.num_classes:
            mask = mask.unsqueeze(-1).repeat(1, 1, self.num_classes)

        pred = pred * mask.float().int()
        target = target * mask.float().int()

        reduction_dims = list(range(len(pred.shape))[1:-1])

        intersection = torch.sum(pred * target, dim=reduction_dims)
        union = torch.sum(pred + target, dim=reduction_dims)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return (1 - torch.mean(dice, dim=-1)).mean()
