# models_parallel_cycle.py
# ------------------------------------------------------------
# Parallelized-per-cycle JointModel:
# - Replace GRUCell + Python for-loop over L with GRU (batch_first=True)
# - Vectorize SOC head (KAN) by flattening [B,L,H] -> [B*L,H]
# - Vectorize CycleEncoder similarly
# - Keep masked attention pooling + SOH KAN head
#
# Result: One cycle forward is parallel over sequence length L on GPU.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1) B-spline basis (degree=3) for 1D input
# ============================================================

def bspline_basis_1d(x: torch.Tensor, knots: torch.Tensor, degree: int = 3) -> torch.Tensor:
    """
    Cox–de Boor recursion for 1D B-spline basis.

    x:     [B]
    knots: [M] non-decreasing
    return: [B, n_basis] where n_basis = len(knots) - degree - 1
    """
    assert x.dim() == 1, f"x must be 1D [B], got {x.shape}"
    device = x.device
    dtype = x.dtype

    knots = knots.to(device=device, dtype=dtype)
    B = x.shape[0]
    x = x.unsqueeze(1)  # [B,1] for broadcasting

    n_basis = len(knots) - degree - 1
    assert n_basis > 0, "Invalid knots/degree resulting in n_basis <= 0"

    M = len(knots)

    # degree 0 basis: N_{i,0}(x) = 1 if knots[i] <= x < knots[i+1]
    N = []
    for i in range(M - 1):
        N.append(((x >= knots[i]) & (x < knots[i + 1])).to(dtype))
    N = torch.cat(N, dim=1)  # [B, M-1]

    # recursion
    for k in range(1, degree + 1):
        N_new = []
        for i in range(M - k - 1):
            denom1 = knots[i + k] - knots[i]
            denom2 = knots[i + k + 1] - knots[i + 1]

            term1 = torch.zeros(B, device=device, dtype=dtype)
            term2 = torch.zeros(B, device=device, dtype=dtype)

            if float(denom1) != 0.0:
                term1 = ((x.squeeze(1) - knots[i]) / denom1) * N[:, i]
            if float(denom2) != 0.0:
                term2 = ((knots[i + k + 1] - x.squeeze(1)) / denom2) * N[:, i + 1]

            N_new.append((term1 + term2).unsqueeze(1))  # [B,1]
        N = torch.cat(N_new, dim=1)  # [B, M-k-1]

    return N[:, :n_basis]  # [B, n_basis]


# ============================================================
# 2) KAN layer with B-spline edges
# ============================================================

class KANLayerBSpline(nn.Module):
    """
    y_j = sum_i sum_m c[j,i,m] * B_m(x_i) + b_j
    """
    def __init__(self, in_dim, out_dim,
                 n_ctrl=8, degree=3,
                 x_min=-1.0, x_max=1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_ctrl = n_ctrl
        self.degree = degree

        # Open-uniform knot vector length = n_ctrl + degree + 1
        knots = torch.linspace(x_min, x_max, n_ctrl + degree + 1)
        self.register_buffer("knots", knots)

        # coeff: [out, in, n_ctrl]
        self.coeff = nn.Parameter(torch.randn(out_dim, in_dim, n_ctrl) * 0.05)
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.x_min = float(x_min)
        self.x_max = float(x_max)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        return: [B, out_dim]
        """
        B, D = x.shape
        assert D == self.in_dim, f"Expected in_dim={self.in_dim}, got {D}"

        x = x.clamp(self.x_min, self.x_max)
        out = torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)

        # loop over input dims (typically small); B dimension is fully parallel
        for i in range(self.in_dim):
            Bi = bspline_basis_1d(x[:, i], self.knots, self.degree)  # [B, n_ctrl]
            out = out + torch.einsum("bm,om->bo", Bi, self.coeff[:, i, :])

        out = out + self.bias.unsqueeze(0)
        return out


class KAN(nn.Module):
    def __init__(self, width, n_ctrl=8, degree=3, x_min=-1.0, x_max=1.0):
        super().__init__()
        layers = []
        for a, b in zip(width[:-1], width[1:]):
            layers.append(
                KANLayerBSpline(a, b, n_ctrl=n_ctrl, degree=degree, x_min=x_min, x_max=x_max)
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = torch.tanh(x)
        return x


# ============================================================
# 3) Attention Pooling (masked)
# ============================================================

class MaskedAttentionPooling(nn.Module):
    """
    z = sum_k alpha_k * h_k
    alpha_k = softmax(score(h_k)) over valid positions
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h_seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        h_seq: [B, L, H]
        mask:  [B, L] (1=valid, 0=pad)  can be float/bool/int
        """
        mask_bool = mask.bool() if mask.dtype != torch.bool else mask

        score = self.score(h_seq).squeeze(-1)      # [B, L]
        neg_inf = torch.finfo(score.dtype).min
        score = score.masked_fill(~mask_bool, neg_inf)
        alpha = torch.softmax(score.float(), dim=1).to(score.dtype)        # [B, L]
        z = torch.sum(h_seq * alpha.unsqueeze(-1), dim=1)  # [B, H]
        return z


# ============================================================
# 4) Parallel SOC / Encoder / SOH (GRU + KAN + Attn Pooling)
# ============================================================

class SOCNetParallel(nn.Module):
    """
    Parallel SOC:
      x_seq [B,L,x_dim] -> GRU -> h_seq [B,L,hidden] -> KAN head -> soc_seq [B,L,1]
    """
    def __init__(self, x_dim=4, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(input_size=x_dim, hidden_size=hidden, batch_first=True)
        self.head = KAN([hidden, 32, 1], n_ctrl=8, degree=3, x_min=-1.0, x_max=1.0)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq: [B, L, x_dim]
        returns soc_seq: [B, L, 1]
        """
        if not x_seq.is_contiguous():
            x_seq = x_seq.contiguous()
        h_seq, _ = self.gru(x_seq)               # [B, L, hidden]
        h_seq = torch.tanh(h_seq)
        B, L, H = h_seq.shape
        soc = self.head(h_seq.reshape(B * L, H)) # [B*L,1]
        soc = torch.sigmoid(soc).view(B, L, 1)
        return soc


class CycleEncoderParallel(nn.Module):
    """
    Parallel encoder:
      u_seq [B,L,in_dim] -> GRU -> h_seq [B,L,hidden]
    """
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.hidden = hidden
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden, batch_first=True)

    def forward(self, u_seq: torch.Tensor) -> torch.Tensor:
        h_seq, _ = self.gru(u_seq)  # [B, L, hidden]
        return h_seq


class SOHHeadKAN(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = KAN([in_dim, 64, 1], n_ctrl=10, degree=3, x_min=-1.0, x_max=1.0)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.tanh(z))


class JointModelParallel(nn.Module):
    """
    Parallelized JointModel:
      - SOC: GRU over full sequence (parallel kernel)
      - Encoder: GRU over full sequence (parallel kernel)
      - Masked attention pooling -> SOH
      - Loss supports masked padded cycles
    """
    def __init__(self, x_dim=4, soc_hidden=64, enc_hidden=128):
        super().__init__()
        self.f1 = SOCNetParallel(x_dim=x_dim, hidden=soc_hidden)
        self.enc = CycleEncoderParallel(in_dim=x_dim + 1, hidden=enc_hidden)
        self.pool = MaskedAttentionPooling(enc_hidden)
        self.f2 = SOHHeadKAN(enc_hidden)

    def forward_one_cycle(self,
                          x_seq: torch.Tensor,              # [B,L,x_dim]
                          soc_gt_seq: torch.Tensor = None,  # [B,L,1] or None
                          soh_gt: torch.Tensor = None,      # [B,1]   or None
                          mask: torch.Tensor = None,        # [B,L]
                          lambda_soc: float = 1.0,
                          lambda_soh: float = 1.0):
        B, L, _ = x_seq.shape
        device = x_seq.device
        if mask is None:
            mask = torch.ones(B, L, device=device, dtype=torch.float32)

        # 1) SOC (parallel over L)
        soc_pred = self.f1(x_seq)                          # [B,L,1]

        # 2) Encoder (parallel over L)
        u_seq = torch.cat([x_seq, soc_pred], dim=-1)       # [B,L,x_dim+1]
        h_seq = self.enc(u_seq)                            # [B,L,H]

        # Apply mask to hidden states (safe for padding)
        h_seq = h_seq * mask.unsqueeze(-1)

        # 3) Pooling -> SOH
        z = self.pool(h_seq, mask)                         # [B,H]
        soh_pred = self.f2(z)                              # [B,1]

        # 4) Loss
        soc_loss = None
        if soc_gt_seq is not None:
            soc_loss = (F.l1_loss(soc_pred, soc_gt_seq, reduction="none")
                        * mask.unsqueeze(-1)).sum() / mask.sum().clamp(min=1.0)

        soh_loss = None
        if soh_gt is not None:
            soh_loss = F.l1_loss(soh_pred, soh_gt)

        total_loss = None
        if (soc_loss is not None) and (soh_loss is not None):
            total_loss = lambda_soc * soc_loss + lambda_soh * soh_loss
        elif soh_loss is not None:
            total_loss = lambda_soh * soh_loss
        elif soc_loss is not None:
            total_loss = lambda_soc * soc_loss

        return soc_pred, soh_pred, total_loss
    
    def forward(self, x_seq, soc_gt_seq=None, soh_gt=None, mask=None,
                lambda_soc: float = 1.0, lambda_soh: float = 1.0):
        # torchinfo / 常规调用都会走这里
        return self.forward_one_cycle(
            x_seq=x_seq,
            soc_gt_seq=soc_gt_seq,
            soh_gt=soh_gt,
            mask=mask,
            lambda_soc=lambda_soc,
            lambda_soh=lambda_soh
        )


# ============================================================
# Optional quick sanity test
# ============================================================

# if __name__ == "__main__":
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model = JointModelParallel(x_dim=4, soc_hidden=64, enc_hidden=128).to(device)

#     B = 3
#     L = 200
#     x = torch.randn(B, L, 4, device=device)
#     mask = torch.ones(B, L, device=device)
#     soc_gt = torch.rand(B, L, 1, device=device)
#     soh_gt = torch.rand(B, 1, device=device)

#     soc_pred, soh_pred, loss = model.forward_one_cycle(x, soc_gt, soh_gt, mask)
#     loss.backward()

#     print("soc_pred:", soc_pred.shape, "soh_pred:", soh_pred.shape, "loss:", float(loss.item()))

# print_model_summary.py
# print_model_summary.py
import torch
from torchinfo import summary

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JointModelParallel(x_dim=4, soc_hidden=64, enc_hidden=128).to(device)

    B, L = 8, 300
    x_seq = torch.randn(B, L, 4, device=device)
    mask  = torch.ones(B, L, device=device)
    soc_gt = torch.rand(B, L, 1, device=device)
    soh_gt = torch.rand(B, 1, device=device)

    summary(
        model,
        input_data=(x_seq, soc_gt, soh_gt, mask),
        col_names=("input_size", "output_size", "num_params", "trainable"),
        depth=4,
        verbose=2
    )


