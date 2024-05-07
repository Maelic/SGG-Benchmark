import torch
from torch import nn
class _SigmoidFocalLoss(nn.Module):
    def __init__(self, gamma: float, alpha: float):
        super(_SigmoidFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        num_classes = logits.shape[1]
        dtype = targets.dtype
        device = targets.device
        class_range = torch.arange(1, num_classes+1, dtype=dtype, device=device).unsqueeze(0)

        t = targets.unsqueeze(1)
        p = torch.sigmoid(logits)
        term1 = (1 - p) ** self.gamma * torch.log(p)
        term2 = p ** self.gamma * torch.log(1 - p)
        loss = -(t == class_range).float() * term1 * self.alpha - ((t != class_range) * (t >= 0)).float() * term2 * (1 - self.alpha)
        return loss.sum()

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "gamma=" + str(self.gamma)
        tmpstr += ", alpha=" + str(self.alpha)
        return tmpstr + ")"

def SigmoidFocalLoss(gamma: float, alpha: float):
    # Create an instance of the class
    loss_fn = _SigmoidFocalLoss(gamma, alpha)

    # Create some example inputs
    logits = torch.randn(1, 4)  # Example logits
    targets = torch.randint(1, 5, (1,))  # Example targets

    # Trace the instance with JIT
    traced_loss_fn = torch.jit.trace(loss_fn, (logits, targets))

    return traced_loss_fn