import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    """
    SwiGLU feedforward block with two linear projections and a SiLU activation.
    """
    def __init__(self, dim, hidden_dim):
        """
        Args:
            dim (int): Input embedding dimension.
            hidden_dim (int): Hidden layer dimension.
        """
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        """
        Forward pass for the SwiGLU feedforward block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, dim).

        Returns:
            torch.Tensor: Output tensor after feedforward transformation.
        """
        return self.w3(self.w1(x) * F.silu(self.w2(x)))
