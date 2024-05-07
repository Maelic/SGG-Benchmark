from sgg_benchmark.layers import SigmoidFocalLoss
import torch

def test_sigmoid_focal_loss():
    # Initialize the loss function
    loss_fn = SigmoidFocalLoss(gamma=2.0, alpha=0.25)

    # Create some sample input tensors
    logits = torch.randn(10, 4)  # 10 samples, 4 classes
    targets = torch.randint(1, 5, (10,))  # 10 samples, classes 1-4

    # Compute the loss
    loss = loss_fn(logits, targets)

    # Check that the loss is a scalar tensor
    assert loss.dim() == 0

    # Check that the loss is non-negative
    assert loss.item() >= 0

# Run the test
test_sigmoid_focal_loss()