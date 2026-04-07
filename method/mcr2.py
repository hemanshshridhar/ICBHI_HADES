import numpy as np
import torch

def one_hot(labels_int, n_classes, device=None, dtype=torch.float32):
    """Turn integer labels into one-hot matrix of shape (N, K)."""
    if not torch.is_tensor(labels_int):
        labels_int = torch.as_tensor(labels_int)
    labels_int = labels_int.long()
    if device is None:
        device = labels_int.device
    labels_int = labels_int.to(device)
    return torch.nn.functional.one_hot(labels_int, num_classes=n_classes).to(dtype)


def label_to_membership(targets, num_classes=None, device=None, dtype=torch.float32):
    """Generate membership tensor Pi of shape (K, N, N) on the same device.

    Pi[k, i, i] = 1 if sample i belongs to class k, else 0.

    Args:
        targets: 1D integer labels (Tensor or array-like), shape (N,)
        num_classes: K
        device: torch device (defaults to targets.device)
        dtype: output dtype

    Returns:
        Pi: Tensor of shape (K, N, N)
    """
    if not torch.is_tensor(targets):
        targets = torch.as_tensor(targets)
    targets = targets.long()
    if device is None:
        device = targets.device
    targets = targets.to(device)

    if num_classes is None:
        # ensure Python int
        num_classes = int(targets.max().item()) + 1

    N = targets.shape[0]
    Pi = torch.zeros((num_classes, N, N), device=device, dtype=dtype)
    idx = torch.arange(N, device=device)
    Pi[targets, idx, idx] = 1.0
    return Pi

class MaximalCodingRateReduction(torch.nn.Module):
    def __init__(self, gam1=1.0, gam2=1.0, eps=0.01):
        super(MaximalCodingRateReduction, self).__init__()
        self.gam1 = gam1
        self.gam2 = gam2
        self.eps = eps

    def compute_discrimn_loss_empirical(self, W):
        """Empirical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + self.gam1 * scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_empirical(self, W, Pi):
        """Empirical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        compress_loss = W.new_tensor(0.0)
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += log_det * trPi / m
        return compress_loss / 2.

    def compute_discrimn_loss_theoretical(self, W):
        """Theoretical Discriminative Loss."""
        p, m = W.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        scalar = p / (m * self.eps)
        logdet = torch.logdet(I + scalar * W.matmul(W.T))
        return logdet / 2.

    def compute_compress_loss_theoretical(self, W, Pi):
        """Theoretical Compressive Loss."""
        p, m = W.shape
        k, _, _ = Pi.shape
        I = torch.eye(p, device=W.device, dtype=W.dtype)
        compress_loss = W.new_tensor(0.0)
        for j in range(k):
            trPi = torch.trace(Pi[j]) + 1e-8
            scalar = p / (trPi * self.eps)
            log_det = torch.logdet(I + scalar * W.matmul(Pi[j]).matmul(W.T))
            compress_loss += trPi / (2 * m) * log_det
        return compress_loss

    def forward(self, X, Y, num_classes=None):
        """Compute MCR^2 loss.

        Expects X as features with shape:
          - (B, D) or
          - (B, L, D) token/sequence features or
          - (B, C, H, W) spatial features
        Converts to (B, D) via mean pooling when needed.
        """
        # Ensure labels are 1D long tensor on same device
        if not torch.is_tensor(Y):
            Y = torch.as_tensor(Y)
        Y = Y.long().to(X.device)

        # Pool / flatten features so X is (B, D)
        if X.dim() == 2:
            X2 = X
        elif X.dim() == 3:
            # (B, L, D) -> mean over L
            X2 = X.mean(dim=1)
        elif X.dim() == 4:
            # (B, C, H, W) -> global average pool over H,W
            X2 = X.mean(dim=(2, 3))
        else:
            # fallback: flatten everything except batch
            X2 = X.reshape(X.shape[0], -1)

        if num_classes is None:
            num_classes = int(Y.max().item()) + 1

        # W should be (D, B)
        W = X2.T
        Pi = label_to_membership(Y, num_classes=num_classes, device=X2.device, dtype=torch.float32)

        discrimn_loss_empi = self.compute_discrimn_loss_empirical(W)
        compress_loss_empi = self.compute_compress_loss_empirical(W, Pi)
        discrimn_loss_theo = self.compute_discrimn_loss_theoretical(W)
        compress_loss_theo = self.compute_compress_loss_theoretical(W, Pi)

        total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
        return (
            total_loss_empi,
            [discrimn_loss_empi.item(), compress_loss_empi.item()],
            [discrimn_loss_theo.item(), compress_loss_theo.item()],
        )
        
'''
usage
criterion = MaximalCodingRateReduction(gam1=1., gam2=1., eps=0.5)        
''' 
