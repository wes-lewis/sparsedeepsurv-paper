from __future__ import annotations
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
#from datasets import load_breast_cancer_data
#from train import train_epoch, evaluate
#from utils import build_affinity_matrix
import numpy as np
from sklearn.neighbors import NearestNeighbors

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
from scipy import sparse

def normalize_adj_selfloop(A: sparse.csr_matrix, add_self_loops: bool = True) -> sparse.csr_matrix:
    """
    Return Â = D^{-1/2} (A + I) D^{-1/2} (if add_self_loops)
    Assumes A is symmetric sparse.
    """
    A = A.tocsr()
    if add_self_loops:
        A = A + sparse.eye(A.shape[0], format="csr")
    d = np.asarray(A.sum(axis=1)).ravel()
    d = np.maximum(d, 1e-12)
    d_inv_sqrt = 1.0 / np.sqrt(d)
    D_inv_sqrt = sparse.diags(d_inv_sqrt, format="csr")
    A_hat = (D_inv_sqrt @ A @ D_inv_sqrt).tocsr()
    return A_hat
class GatedGeneToModule(nn.Module):
    """
    Global gated dictionary:
      W = M ⊙ B
    where:
      B: (G,K) real weights
      M: (G,K) near-binary mask from Concrete/LSPIN gates (global parameters)

    Forward:
      z = x_tilde @ W   (B,G) @ (G,K) -> (B,K)

    Returns:
      z, dict_gate_probs   where dict_gate_probs is the expected activation of M (G,K)
    """
    def __init__(
        self,
        n_genes: int,
        n_modules: int,
        gate_type: str = "concrete",   # "concrete", "lspin", "lspin_tf"
        temperature: float = 0.5,
        gate_sigma: float = 1.0,
        a: float = 1.0,
        init_mask_bias: float = -2.0,  # sparsity-init for logits/alpha
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_modules = int(n_modules)

        self.gate_type = str(gate_type).lower()
        self.temperature = float(temperature)
        self.gate_sigma = float(gate_sigma)
        self.a = float(a)

        # real weights
        self.B = nn.Parameter(torch.empty(n_genes, n_modules))
        nn.init.xavier_uniform_(self.B)

        # global gate parameters
        if self.gate_type == "concrete":
            self.logits = nn.Parameter(torch.full((n_genes, n_modules), float(init_mask_bias)))
        elif self.gate_type in ("lspin", "lspin_tf"):
            # alpha plays role of gating_net output; global instead of per-sample
            self.alpha = nn.Parameter(torch.full((n_genes, n_modules), float(init_mask_bias)))
        else:
            raise ValueError(f"Unknown gate_type for dictionary: {self.gate_type}")

    def _sample_concrete(self, logits):
        u = torch.rand_like(logits)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        gate_sample = torch.sigmoid((logits + gumbel) / self.temperature)
        gate_probs = torch.sigmoid(logits)
        return gate_sample, gate_probs

    def _sample_lspin(self, alpha):
        # match your per-sample LSPIN: mu=tanh(alpha), z=clip(0.5+mu+eps)
        mu = torch.tanh(alpha)
        eps = torch.randn_like(mu) * self.gate_sigma
        gate_sample = torch.clamp(0.5 + mu + eps, 0.0, 1.0)
        gate_probs = 0.5 * (1.0 + torch.erf((mu + 0.5) / (math.sqrt(2.0) * self.gate_sigma + 1e-8)))
        return gate_sample, gate_probs

    def _sample_lspin_tf(self, alpha):
        # TF-faithful: hard_sigmoid(a*alpha+0.5) + Gaussian noise
        g_det = torch.clamp(self.a * alpha + 0.5, 0.0, 1.0)
        eps = torch.randn_like(alpha)
        z = alpha + self.gate_sigma * eps
        gate_sample = torch.clamp(self.a * z + 0.5, 0.0, 1.0)

        # TF reg prob surrogate (same as your _sample_lspin_tf)
        reg_prob = 0.5 - 0.5 * torch.erf(
            (-1.0 / (2.0 * self.a) - alpha) / (self.gate_sigma * math.sqrt(2.0) + 1e-8)
        )
        return gate_sample, g_det, reg_prob

    def forward(self, x_tilde: torch.Tensor):
        """
        x_tilde: (B,G)
        returns:
          z: (B,K)
          dict_gate_probs: (G,K)  expected activation probability for mask M
        """
        if self.gate_type == "concrete":
            M_samp, M_prob = self._sample_concrete(self.logits)
            M_det = torch.sigmoid(self.logits)
            dict_gate_probs = M_prob
            self._last_M_det = M_det.detach()

        elif self.gate_type == "lspin":
            M_samp, M_prob = self._sample_lspin(self.alpha)
            M_det = torch.clamp(0.5 + torch.tanh(self.alpha), 0.0, 1.0)
            dict_gate_probs = M_prob
            self._last_M_det = M_det.detach()

        elif self.gate_type == "lspin_tf":
            M_samp, M_det, reg_prob = self._sample_lspin_tf(self.alpha)
            dict_gate_probs = reg_prob
            self._last_M_det = M_det.detach()

        else:
            raise ValueError(self.gate_type)

        W = M_samp * self.B   # (G,K)
        z = x_tilde @ W       # (B,K)
        return z, dict_gate_probs

def scipy_csr_to_torch_sparse(A: sparse.csr_matrix, device="cpu", dtype=torch.float32) -> torch.Tensor:
    """
    Convert scipy CSR/COO to torch sparse COO tensor.
    """
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx, val, size=A.shape, device=device).coalesce()

class GeneGraphSmoother(nn.Module):
    """
    Applies x_tilde = Â x for a batch.
    x: (B, G)
    Â: torch sparse (G, G)
    """
    def __init__(self, A_hat_torch_sparse: torch.Tensor):
        super().__init__()
        if not A_hat_torch_sparse.is_sparse:
            raise ValueError("A_hat_torch_sparse must be a torch sparse tensor")
        self.register_buffer("A_hat", A_hat_torch_sparse.coalesce())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (G,G) @ (G,B) -> (G,B) -> (B,G)
        return torch.sparse.mm(self.A_hat, x.t()).t()
class GeneToModuleLinear(nn.Module):
    """
    z = x_tilde @ W   (B, G) @ (G, K) -> (B, K)
    """
    def __init__(self, n_genes: int, n_modules: int, col_norm: bool = True):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_genes, n_modules))
        nn.init.xavier_uniform_(self.W)
        self.col_norm = bool(col_norm)

    def forward(self, x_tilde: torch.Tensor) -> torch.Tensor:
        W = self.W
        if self.col_norm:
            # normalize columns to stabilize interpretation (optional)
            W = W / (W.norm(dim=0, keepdim=True) + 1e-8)
        return x_tilde @ W  # (B, K)

    def l1_penalty(self, lambda_l1: float) -> torch.Tensor:
        return float(lambda_l1) * self.W.abs().sum()
class DeepSurvModuleGatedDictSparse(nn.Module):
    """
    x -> graph smoothing -> gated gene->module dictionary -> module LSPIN gate -> Cox MLP

    forward returns:
      risk, module_gate_probs
    and stores:
      self._last_dict_gate_probs (G,K) for the trainer to penalize
    """
    def __init__(
        self,
        n_genes: int,
        n_modules: int,
        A_hat_torch_sparse: torch.Tensor,
        dict_gate_type: str = "concrete",
        dict_temperature: float = 0.5,
        dict_gate_sigma: float = 1.0,
        dict_a: float = 1.0,
        dict_init_alpha_bias: float = -2.0,      # NEW (used for lspin_tf dict)
        module_gate_type: str = "lspin_tf",
        module_temperature: float = 0.5,
        module_gate_sigma: float = 0.5,
        module_a: float = 1.0,
        module_init_alpha_bias: float = -2.0,    # NEW (used for lspin_tf module)
        gating_hidden_dim: int = 128,
        dropout_p: float = 0.0,
        mlp_hidden_dims=(64, 32),
    ):
        super().__init__()
        self.smoother = GeneGraphSmoother(A_hat_torch_sparse)

        self.dict = GatedGeneToModule(
            n_genes=n_genes,
            n_modules=n_modules,
            gate_type=dict_gate_type,
            temperature=dict_temperature,
            gate_sigma=dict_gate_sigma,
            a=dict_a,
            init_mask_bias=dict_init_alpha_bias,   # CHANGED: use arg
        )

        # module gate (instance-wise)
        self.temperature = float(module_temperature)
        self.dropout_p = float(dropout_p)
        self.gate_type = str(module_gate_type).lower()
        self.gate_sigma = float(module_gate_sigma)
        self.a = float(module_a)

        self.gating_net = nn.Sequential(
            nn.Linear(n_modules, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, n_modules),
        )

        # IMPORTANT: initialize module gate bias ONCE (not in forward)
        if self.gate_type == "lspin_tf":
            last = self.gating_net[-1]
            assert isinstance(last, nn.Linear)
            with torch.no_grad():
                last.bias.fill_(float(module_init_alpha_bias))

        # Cox MLP head
        layers = []
        prev = n_modules
        for h in mlp_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout_p > 0:
                layers += [nn.Dropout(dropout_p)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.risk_net = nn.Sequential(*layers)

    # module gate samplers (reuse your code)
    def _sample_concrete(self, logits):
        u = torch.rand_like(logits)
        gate_probs = torch.sigmoid(logits)
        gumbel = -torch.log(-torch.log(u + 1e-8) + 1e-8)
        gate_sample = torch.sigmoid((logits + gumbel) / self.temperature)
        return gate_sample, gate_probs

    def _sample_lspin(self, z):
        mu_raw = self.gating_net(z)
        mu = torch.tanh(mu_raw)
        eps = torch.randn_like(mu) * self.gate_sigma
        gate_sample = torch.clamp(0.5 + mu + eps, 0.0, 1.0)
        gate_probs = 0.5 * (1.0 + torch.erf((mu + 0.5) / (math.sqrt(2.0) * self.gate_sigma + 1e-8)))
        return gate_sample, gate_probs

    def _sample_lspin_tf(self, z):
        alpha = self.gating_net(z)
        g_det = torch.clamp(self.a * alpha + 0.5, 0.0, 1.0)
        eps = torch.randn_like(alpha)
        z_noisy = alpha + self.gate_sigma * eps
        gate_sample = torch.clamp(self.a * z_noisy + 0.5, 0.0, 1.0)

        reg_prob = 0.5 - 0.5 * torch.erf(
            (-1.0 / (2.0 * self.a) - alpha) / (self.gate_sigma * math.sqrt(2.0) + 1e-8)
        )
        return gate_sample, g_det, reg_prob

    def forward(self, x, deterministic: bool = False):
        x_tilde = self.smoother(x)          # (B,G)

        z, dict_gate_probs = self.dict(x_tilde)  # z: (B,K), dict_gate_probs: (G,K)
        self._last_dict_gate_probs = dict_gate_probs  # stash for trainer

        # instance-wise module gating
        if self.gate_type == "concrete":
            logits = self.gating_net(z)
            gate_sample, gate_probs = self._sample_concrete(logits)
            g_det_cont = torch.sigmoid(logits)
        elif self.gate_type == "lspin":
            gate_sample, gate_probs = self._sample_lspin(z)
            mu_raw = self.gating_net(z)
            mu = torch.tanh(mu_raw)
            g_det_cont = torch.clamp(0.5 + mu, 0.0, 1.0)
        elif self.gate_type == "lspin_tf":
            gate_sample, g_det, reg_prob = self._sample_lspin_tf(z)
            g_det_cont = g_det
            gate_probs = reg_prob
        else:
            raise ValueError(self.gate_type)

        # Optional deterministic inference path (paper-consistent).
        if deterministic:
            gate_sample = g_det_cont

        self._last_g_det_cont = g_det_cont.detach()
        if (not deterministic) and self.dropout_p > 0:
            gate_sample = F.dropout(gate_sample, p=self.dropout_p, training=self.training)

        z_gated = z * gate_sample
        risk = self.risk_net(z_gated).squeeze(-1)
        risk = torch.clamp(risk, min=-20, max=20)
        return risk, gate_probs

import torch, numpy as np

@torch.no_grad()
def module_activations_and_gates(model, X, device="cpu"):
    """
    Returns:
      z: (N,K) module activations (pre-module-gate)
      g_det: (N,K) deterministic module gates in [0,1]
    """
    model.eval()
    X = X.to(device)

    x_tilde = model.smoother(X)  # (N,G)

    dict_out = model.dict(x_tilde)
    # dict_out can be z or (z, dict_gate_probs)
    if isinstance(dict_out, (tuple, list)):
        z = dict_out[0]
    else:
        z = dict_out

    # deterministic module gate depends on module gate type
    if model.gate_type == "lspin_tf":
        alpha = model.gating_net(z)
        g_det = torch.clamp(model.a * alpha + 0.5, 0.0, 1.0)
    elif model.gate_type == "lspin":
        mu = torch.tanh(model.gating_net(z))
        g_det = torch.clamp(0.5 + mu, 0.0, 1.0)
    else:  # concrete
        logits = model.gating_net(z)
        g_det = torch.sigmoid(logits)

    return z.detach().cpu(), g_det.detach().cpu()
@torch.no_grad()
def dict_mask_det(model):
    """
    Returns deterministic dictionary mask M_det of shape (G,K) in [0,1].
    Works for the gated dictionary version (Concrete/LSPIN/LSPIN_TF).
    """
    d = model.dict
    gt = getattr(d, "gate_type", None)

    if gt is None:
        raise ValueError("model.dict has no gate_type; are you using the gated dictionary class?")

    gt = gt.lower()
    if gt == "concrete":
        return torch.sigmoid(d.logits).detach().cpu()          # (G,K)
    elif gt == "lspin":
        return torch.clamp(0.5 + torch.tanh(d.alpha), 0, 1).detach().cpu()
    elif gt == "lspin_tf":
        return torch.clamp(d.a * d.alpha + 0.5, 0, 1).detach().cpu()
    else:
        raise ValueError(f"Unknown dict gate_type: {gt}")

@torch.no_grad()
def genes_in_module(model, module_k, gene_names, thr=0.5, top_n=None):
    """
    Return gene names in module k using deterministic dict mask.
    If top_n is set, returns top_n genes by mask strength.
    """
    M = dict_mask_det(model).numpy()  # (G,K)
    m = M[:, module_k]

    if top_n is not None:
        idx = np.argsort(-m)[:top_n]
    else:
        idx = np.where(m > thr)[0]

    return list(gene_names[idx]), idx, m[idx]
@torch.no_grad()
def top_genes_for_patient_module_gateddict(model, x_patient, module_k, gene_names, top_n=30, device="cpu"):
    """
    Returns top genes (by absolute contribution) to module_k for one patient.
    """
    model.eval()
    x_patient = x_patient.to(device).view(1, -1)   # (1,G)
    x_tilde = model.smoother(x_patient).view(-1)   # (G,)

    # deterministic mask for dict
    M_det = dict_mask_det(model).to(device)        # (G,K)
    m_k = M_det[:, module_k]                       # (G,)

    # real weights
    B = model.dict.B.detach()                      # (G,K)
    b_k = B[:, module_k]

    w_eff = m_k * b_k                              # (G,)
    contrib = (x_tilde * w_eff).detach().cpu().numpy()

    idx = np.argsort(-np.abs(contrib))[:top_n]
    out = [(gene_names[i], float(contrib[i]), float(m_k[i].detach().cpu()), float(b_k[i].detach().cpu()), float(x_tilde[i].detach().cpu()))
           for i in idx]
    return out


class DeepSurvModuleGated(nn.Module):
    """
    x (B,G)
      -> x_tilde = Â x
      -> z = x_tilde @ W  (B,K)
      -> gate over modules: g(x) in [0,1]^K (LSPIN/Concrete)
      -> z_gated = z * g
      -> Cox MLP on modules
    """
    def __init__(
        self,
        n_genes: int,
        n_modules: int,
        A_hat_torch_sparse: torch.Tensor,
        hidden_dim: int = 64,
        gating_hidden_dim: int = 128,
        gate_type: str = "lspin_tf",   # reuse your implementations
        temperature: float = 0.5,
        gate_sigma: float = 1.0,
        a: float = 1.0,
        dropout_p: float = 0.0,
        mlp_hidden_dims=(64, 32),
        dict_col_norm: bool = True,
    ):
        super().__init__()
        self.n_genes = int(n_genes)
        self.n_modules = int(n_modules)

        self.smoother = GeneGraphSmoother(A_hat_torch_sparse)
        self.dict = GeneToModuleLinear(n_genes, n_modules, col_norm=dict_col_norm)

        # gating net operates on module activations z (B,K) and outputs (B,K)
        self.temperature = float(temperature)
        self.dropout_p = float(dropout_p)
        self.gate_type = str(gate_type).lower()
        self.gate_sigma = float(gate_sigma)
        self.a = float(a)

        self.gating_net = nn.Sequential(
            nn.Linear(n_modules, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, n_modules),
        )

        # Cox MLP head on gated modules
        layers = []
        prev = n_modules
        for h in mlp_hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout_p > 0:
                layers += [nn.Dropout(dropout_p)]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.risk_net = nn.Sequential(*layers)

        if self.gate_type == "lspin_tf":
            last = self.gating_net[-1]
            with torch.no_grad():
                last.bias.fill_(-2.0)

    # ---- reuse gating from your DeepSurvGated ----
    def _sample_concrete(self, logits):
        uniform = torch.rand_like(logits)
        gate_probs = torch.sigmoid(logits)
        gumbel = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        gate_sample = torch.sigmoid((logits + gumbel) / self.temperature)
        return gate_sample, gate_probs

    def _sample_lspin_tf(self, z):
        alpha = self.gating_net(z)
        g_det = torch.clamp(self.a * alpha + 0.5, 0.0, 1.0)
        eps = torch.randn_like(alpha)
        z_noisy = alpha + self.gate_sigma * eps
        gate_sample = torch.clamp(self.a * z_noisy + 0.5, 0.0, 1.0)

        reg_prob = 0.5 - 0.5 * torch.erf(
            (-1.0 / (2.0 * self.a) - alpha) / (self.gate_sigma * math.sqrt(2.0) + 1e-8)
        )
        return gate_sample, g_det, reg_prob

    def _sample_lspin(self, z):
        mu_raw = self.gating_net(z)
        mu = torch.tanh(mu_raw)
        eps = torch.randn_like(mu) * self.gate_sigma
        gate_sample = torch.clamp(0.5 + mu + eps, 0.0, 1.0)
        gate_probs = 0.5 * (1.0 + torch.erf((mu + 0.5) / (math.sqrt(2.0) * self.gate_sigma + 1e-8)))
        return gate_sample, gate_probs

    def forward(self, x, deterministic: bool = False):
        """
        Returns:
          risk: (B,)
          gate_probs: (B,K) expected activation of modules (for sparsity penalty etc)
        """
        x_tilde = self.smoother(x)           # (B,G)
        z = self.dict(x_tilde)               # (B,K)

        if self.gate_type == "concrete":
            logits = self.gating_net(z)
            gate_sample, gate_probs = self._sample_concrete(logits)
            g_det_cont = torch.sigmoid(logits)

        elif self.gate_type == "lspin":
            gate_sample, gate_probs = self._sample_lspin(z)
            mu_raw = self.gating_net(z)
            mu = torch.tanh(mu_raw)
            g_det_cont = torch.clamp(0.5 + mu, 0.0, 1.0)

        elif self.gate_type == "lspin_tf":
            gate_sample, g_det, reg_prob = self._sample_lspin_tf(z)
            g_det_cont = g_det
            gate_probs = reg_prob

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")

        # Optional deterministic inference path (paper-consistent).
        if deterministic:
            gate_sample = g_det_cont

        self._last_g_det_cont = g_det_cont.detach()

        if (not deterministic) and self.dropout_p > 0:
            gate_sample = F.dropout(gate_sample, p=self.dropout_p, training=self.training)
        z_gated = z * gate_sample

        risk = self.risk_net(z_gated).squeeze(-1)
        risk = torch.clamp(risk, min=-20, max=20)

        return risk, gate_probs

    def dict_l1_penalty(self, lambda_l1_dict: float) -> torch.Tensor:
        return self.dict.l1_penalty(lambda_l1_dict)


class DeepSurvGated(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        gating_hidden_dim: int,
        temperature: float = 0.5,
        llspin: bool = False,
        dropout_p: float = 0.0,
        gate_type: str = "concrete",   # "concrete", "lspin", "lspin_tf"
        gate_sigma: float = 1.0,       # σ
        a: float = 1.0,                # hard-sigmoid slope used in TF LSPIN
    ):
        super().__init__()
        self.llspin = bool(llspin)
        self.temperature = float(temperature)
        self.dropout_p = float(dropout_p)
        self.gate_type = str(gate_type).lower()

        # parameters needed for TF-faithful gating
        self.a = float(a)
        self.gate_sigma = float(gate_sigma)

        # Gating network
        self.gating_net = nn.Sequential(
            nn.Linear(input_dim, gating_hidden_dim),
            nn.ReLU(),
            nn.Linear(gating_hidden_dim, input_dim),
        )

        # Risk network
        if self.llspin:
            self.risk_net = nn.Linear(input_dim, 1)
        else:
            self.risk_net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        if self.gate_type == "lspin_tf":
            last = self.gating_net[-1]
            if not isinstance(last, nn.Linear):
                raise TypeError("Expected gating_net[-1] to be nn.Linear")
            with torch.no_grad():
                last.bias.fill_(-2.0)
        
            


    # ---------- Gating implementations ----------

    def _sample_concrete(self, logits):
        """
        Concrete (Gumbel–Sigmoid) gates:
            gate_sample ~ Concrete(sigmoid(logits), temperature)
            gate_probs  = sigmoid(logits)
        """
        uniform = torch.rand_like(logits)
        gate_probs = torch.sigmoid(logits)

        gumbel = -torch.log(-torch.log(uniform + 1e-8) + 1e-8)
        gate_sample = torch.sigmoid((logits + gumbel) / self.temperature)

        return gate_sample, gate_probs

    def _sample_lspin_tf(self, x):
        alpha = self.gating_net(x)
    
        # deterministic gates: hard_sigmoid(alpha)
        g_det = torch.clamp(self.a * alpha + 0.5, 0.0, 1.0)
    
        # stochastic gates used for masking during training
        eps = torch.randn_like(alpha)
        z = alpha + self.gate_sigma * eps
        gate_sample = torch.clamp(self.a * z + 0.5, 0.0, 1.0)
    
        # TF L0 surrogate probability used in reg term
        reg_prob = 0.5 - 0.5 * torch.erf(
            (-1.0 / (2.0 * self.a) - alpha) / (self.gate_sigma * math.sqrt(2.0) + 1e-8)
        )
    
        return gate_sample, g_det, reg_prob

    def _sample_lspin(self, x):
        """
        LSPIN-style Gaussian-threshold gates.

        Paper definition:
            z = clip(0.5 + µ + ε, 0, 1),  ε ~ N(0, σ^2)

        Here:
            - µ = tanh(gating_net(x))  (kept in [-1, 1] as in the paper)
            - gate_sample = z
            - gate_probs ≈ P[z > 0]
              = Φ((µ + 0.5)/σ) = 0.5 * (1 + erf((µ + 0.5)/(σ * sqrt(2))))
        """
        # Predict µ and squash to [-1, 1] (the paper uses tanh in the gating net)
        mu_raw = self.gating_net(x)
        mu = torch.tanh(mu_raw)

        # Sample Gaussian noise
        eps = torch.randn_like(mu) * self.gate_sigma

        # Hard-threshold gate (stochastic during training)
        z = 0.5 + mu + eps
        gate_sample = torch.clamp(z, 0.0, 1.0)

        # Expected activation probability P(z > 0)
        gate_probs = 0.5 * (
            1.0
            + torch.erf(
                (mu + 0.5) / (math.sqrt(2.0) * self.gate_sigma + 1e-8)
            )
        )

        return gate_sample, gate_probs

    # ---------- Forward ----------

    def forward(self, x, deterministic: bool = False):
        """
        Returns
        -------
        risk : (N,)
            Cox log-risk.
        gate_probs : (N, D)
            Used for sparsity penalty and smoothness losses.
            For lspin_tf, gate_probs == reg_prob (TF-style L0 surrogate).
        """
        if self.gate_type == "concrete":
            gate_logits = self.gating_net(x)
            gate_sample, gate_probs = self._sample_concrete(gate_logits)
    
            # deterministic continuous gate for reporting
            g_det_cont = torch.sigmoid(gate_logits)
    
        elif self.gate_type == "lspin":
            gate_sample, gate_probs = self._sample_lspin(x)
    
            # deterministic (noise-free) version: z = clip(0.5 + tanh(alpha), 0, 1)
            mu_raw = self.gating_net(x)
            mu = torch.tanh(mu_raw)
            g_det_cont = torch.clamp(0.5 + mu, 0.0, 1.0)
    
        elif self.gate_type == "lspin_tf":
            gate_sample, g_det, reg_prob = self._sample_lspin_tf(x)
    
            # TF deterministic hard-sigmoid already computed
            g_det_cont = g_det
    
            # for the trainer: gate_probs should be reg_prob (TF reg uses this)
            gate_probs = reg_prob

        else:
            raise ValueError(f"Unknown gate_type: {self.gate_type}")
    
        # Optional deterministic inference path (paper-consistent).
        if deterministic:
            gate_sample = g_det_cont

        # store deterministic continuous gates for logging / K_hard
        self._last_g_det_cont = g_det_cont.detach()
    
        # Optionally drop gates during training
        if (not deterministic) and self.dropout_p > 0:
            gate_sample = F.dropout(gate_sample, p=self.dropout_p, training=self.training)
    
        # Elementwise mask inputs
        gated_x = x * gate_sample
    
        # Risk score
        risk = self.risk_net(gated_x).squeeze(-1)
        risk = torch.clamp(risk, min=-20, max=20)
    
        return risk, gate_probs


def sparsity_loss(gate_probs, lambda_sparse):
    return lambda_sparse * gate_probs.mean()
def smoothness_loss(gate_probs, affinity_matrix, lambda_smooth):
    """
    gate_probs: (N, F)
    affinity_matrix: (N, N)
    """
    diff = gate_probs.unsqueeze(1) - gate_probs.unsqueeze(0)  # (N, N, F)
    sq_norm = (diff**2).sum(-1)
    smoothness = (affinity_matrix * sq_norm).mean()
    return lambda_smooth * smoothness
def negative_partial_log_likelihood(risk, durations, events):
    """
    risk: tensor (N,)
    durations: tensor (N,)
    events: tensor (N,)
    """
    # Sort by descending durations
    order = torch.argsort(-durations)
    risk = risk[order]
    events = events[order]
    
    # Compute the cumulative log-sum-exp denominator over the risk set
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    
    # Only include terms for events==1
    likelihood = (risk - log_cumsum) * events
    
    return -likelihood.sum() / events.sum()


def compute_knn_affinity(X, k=10): #cellular affinity
    """
    X: numpy array (N, F)
    Returns: sparse csr_matrix (N, N) with 1 for neighbors, 0 otherwise
    """
    nbrs = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(X)
    knn_graph = nbrs.kneighbors_graph(X, mode="connectivity")
    # Symmetrize to ensure mutual neighbors
    affinity = knn_graph.maximum(knn_graph.transpose())
    return affinity

import numpy as np
from scipy import sparse

def compute_gene_affinity(X, bandwidth=0.1, k=None, min_affinity=None):
    """
    Computes gene-gene affinity matrix from expression data.

    Parameters:
    ----------
    X : np.ndarray
        (N_samples, N_genes)
    bandwidth : float
        Controls sharpness of affinity decay (smaller = sharper).
    k : int or None
        If set, keep only top-k affinities per gene.
    min_affinity : float or None
        If set, zero out affinities below this threshold.

    Returns:
    -------
    affinity_sparse : scipy.sparse.csr_matrix
        Sparse affinity matrix (N_genes x N_genes)
    """
    N, G = X.shape

    # Compute Pearson correlations between genes
    corr = np.corrcoef(X.T)  # shape (G,G)
    corr = np.nan_to_num(corr, nan=0.0)  # in case any constant columns

    # Take absolute value
    abs_corr = np.abs(corr)

    # Gaussian-like affinity
    affinity_dense = np.exp(-(1 - abs_corr) / bandwidth)

    # Optionally threshold
    if min_affinity is not None:
        affinity_dense[affinity_dense < min_affinity] = 0.0

    # Optionally keep top-k per gene
    if k is not None:
        rows = []
        cols = []
        data = []
        for i in range(G):
            row = affinity_dense[i, :]
            # Get indices of top-k (excluding self if you want)
            top_idx = np.argpartition(-row, k+1)[:k+1]  # +1 in case self-affinity
            for j in top_idx:
                if i == j:
                    continue  # skip self if desired
                rows.append(i)
                cols.append(j)
                data.append(row[j])
        affinity_sparse = sparse.csr_matrix((data, (rows, cols)), shape=(G, G))
    else:
        affinity_sparse = sparse.csr_matrix(affinity_dense)

    # Symmetrize
    affinity_sparse = affinity_sparse.maximum(affinity_sparse.T)

    return affinity_sparse

import numpy as np
from scipy import sparse

def smoothness_loss_sparse(gate_probs, affinity_matrix, lambda_smooth):
    """
    gate_probs: (N, F) Tensor on device
    affinity_matrix: scipy sparse CSR matrix
    """
    coo = affinity_matrix.tocoo()
    i_idx = torch.tensor(coo.row, device=gate_probs.device)
    j_idx = torch.tensor(coo.col, device=gate_probs.device)

    diffs = gate_probs[i_idx] - gate_probs[j_idx]  # (nnz, F)
    sq_norm = (diffs**2).sum(-1)  # (nnz,)

    smooth = sq_norm.mean()
    return lambda_smooth * smooth

def plot_clustered_gating_masks(
    z_tensor,
    sample_labels=None,
    feature_labels=None,
    tumor_types=None,
    selected_indices=None,
    activation_threshold=0.1,
    title="Gating masks with hierarchical clustering"
):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    z_np = z_tensor.cpu().numpy()

    # Compute mean activation per gene
    mean_activations = z_np.mean(axis=0)
    keep_mask = mean_activations > activation_threshold
    z_filtered = z_np[:, keep_mask]

    if feature_labels is not None:
        feature_labels_filtered = np.array(feature_labels)[keep_mask]
    else:
        feature_labels_filtered = None

    if z_filtered.shape[1] == 0:
        print("No genes passed the activation threshold.")
        return None

    df = pd.DataFrame(
        z_filtered,
        index=sample_labels if sample_labels is not None else None,
        columns=feature_labels_filtered
    )

    # Tumor type colors
    row_colors = None
    lut = None
    if tumor_types is not None:
        tumor_types_series = pd.Series(tumor_types, index=df.index)
        unique_types = tumor_types_series.unique()
        palette = sns.color_palette("tab20", n_colors=len(unique_types))
        lut = dict(zip(unique_types, palette))
        row_colors = tumor_types_series.map(lut)

    # Clustermap
    g = sns.clustermap(
        df,
        method="average",
        metric="correlation",
        cmap="viridis",
        figsize=(max(20, df.shape[1] // 2), 25),
        cbar_pos=None,   # Remove the colorbar
        dendrogram_ratio=(0.05, 0.1),  # Narrower left dendrogram
        yticklabels=True if sample_labels is not None else False,
        xticklabels=True if feature_labels_filtered is not None else True,
        row_colors=row_colors
    )

    # Rotate and resize labels
    for label in g.ax_heatmap.get_xticklabels():
        label.set_rotation(90)
        label.set_fontsize(18)
    for label in g.ax_heatmap.get_yticklabels():
        label.set_fontsize(14)

    # Highlight selected indices
    if selected_indices is not None:
        selected_indices_set = set(
            selected_indices.numpy() if hasattr(selected_indices, "numpy") else selected_indices
        )
        reordered_indices = g.dendrogram_row.reordered_ind
        y_ticklabels = g.ax_heatmap.get_yticklabels()
        for tick, sample_idx in zip(y_ticklabels, reordered_indices):
            if sample_idx in selected_indices_set:
                tick.set_color("blue")
                tick.set_fontweight("bold")

    # Figure-level title with controlled spacing
    g.fig.suptitle(title, fontsize=18, y=.98)

    # Tumor type legend
    if tumor_types is not None and lut:
        handles = [Patch(facecolor=lut[label], edgecolor='k', label=label) for label in lut]
        g.ax_heatmap.legend(
            handles,
            lut.keys(),
            title="Tumor Type",
            loc="upper left",
            bbox_to_anchor=(1.05, 1.0)
        )

    # Tight layout
    g.fig.tight_layout(rect=[0, 0, 1, 0.98])
    plt.show()

    return g


def plot_expression_matching_gating(
    expression_tensor,
    clustering_g,
    tumor_types=None,
    sample_labels=None,
    zscore=True,
    title="Expression values aligned to gating"
):
    """
    expression_tensor: torch tensor (num_samples, num_genes)
    clustering_g: sns.clustermap object from gating masks
    tumor_types: array-like of tumor types (aligned with sample_labels)
    sample_labels: list/array of sample IDs (must match order of expression_tensor)
    """
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    expression_np = expression_tensor.cpu().numpy()

    # Get gene order
    ordered_gene_names = np.array(clustering_g.data2d.columns)
    gene_idx_map = {name: i for i, name in enumerate(gene_names)}
    selected_gene_indices = [gene_idx_map[name] for name in ordered_gene_names]

    # Get sample IDs in order
    ordered_sample_ids = clustering_g.data2d.index.to_numpy()

    # Build DataFrame matching sample IDs
    expr_df = pd.DataFrame(
        expression_np,
        index=sample_labels,
        columns=gene_names
    )

    # Subset to same genes and same samples
    df_subset = expr_df.loc[ordered_sample_ids, ordered_gene_names]

    # Z-score
    if zscore:
        df_subset = (df_subset - df_subset.mean(axis=0)) / (df_subset.std(axis=0) + 1e-6)

    # Tumor colors
    row_colors = None
    lut = None
    if tumor_types is not None:
        tumor_series = pd.Series(tumor_types, index=sample_labels)
        tumor_series = tumor_series.loc[ordered_sample_ids]
        unique_types = pd.unique(tumor_series)
        palette = sns.color_palette("tab20", n_colors=len(unique_types))
        lut = dict(zip(unique_types, palette))
        row_colors = tumor_series.map(lut)

    # Plot without clustering and no colorbar
    g_expr = sns.clustermap(
        df_subset,
        row_cluster=False,
        col_cluster=False,
        cmap="RdBu_r",
        center=0,
        figsize=(max(20, df_subset.shape[1] // 2.5), 20),
        yticklabels=True,
        xticklabels=True,
        row_colors=row_colors,
        cbar_pos=None,
    )

    # Remove dendrogram axes
    g_expr.ax_row_dendrogram.set_visible(False)
    g_expr.ax_col_dendrogram.set_visible(False)

    # Adjust tick labels
    plt.setp(g_expr.ax_heatmap.get_xticklabels(), rotation=90, fontsize=18)
    plt.setp(g_expr.ax_heatmap.get_yticklabels(), fontsize=12)

    # Title placed directly over heatmap (avoids extra top space)
    g_expr.ax_heatmap.set_title(title, fontsize=16, pad=10)

    # Tumor-type categorical legend
    if tumor_types is not None and lut:
        handles = [Patch(facecolor=lut[label], edgecolor='k', label=label) for label in lut]
        g_expr.ax_heatmap.legend(
            handles,
            lut.keys(),
            title="Tumor Type",
            loc="upper left",
            bbox_to_anchor=(1.05, 1.0)
        )

    # Tight layout
    g_expr.fig.tight_layout()
    plt.show()
    return g_expr

class CoxLinear(nn.Module):
    """
    Global linear Cox model: η(x) = β^T x + b
    Use L2 via optimizer weight_decay, and/or L1 via explicit penalty.
    """
    def __init__(self, input_dim: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1, bias=bias)

    def forward(self, x):
        # x: (N, D)
        risk = self.linear(x).squeeze(-1)  # (N,)
        # Optional: clamp for numerical stability
        return torch.clamp(risk, min=-20, max=20)

    
#L2 (Ridge) example
#model = CoxLinear(input_dim=X_train.shape[1])
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

#L1 (Lasso) example
#lambda_l1 = 1e-3
#beta = model.linear.weight  # shape (1, D)
#lasso_penalty = lambda_l1 * beta.abs().sum()
#loss = npll + lasso_penalty

#This is DeepSurv without gating
class DeepSurvMLP(nn.Module):
    """
    DeepSurv-style MLP Cox model.
    - L2 via optimizer weight_decay
    - L1 via explicit penalty on input layer weights
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims=(64, 32),
        activation=nn.ReLU,
        dropout_p: float = 0.0,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation())
            if dropout_p > 0:
                layers.append(nn.Dropout(dropout_p))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

        # Keep a reference to the first linear layer for L1 feature sparsity
        self.input_layer = self.net[0]  # nn.Linear(input_dim, hidden_dims[0])

    def forward(self, x):
        risk = self.net(x).squeeze(-1)
        return torch.clamp(risk, min=-20, max=20)

    #L2 is again accomplished with weight_decay in optimizer
#L1 example
#lambda_l1_input = 1e-3
#W_in = model.input_layer.weight  # (hidden_dim, D)
#lasso_input = lambda_l1_input * W_in.abs().sum()
#loss = npll + lasso_input
def train_deepsurv_mlp_group_lasso(
    model,
    X_train, time_train, event_train,
    X_val, time_val, event_val,
    lr=1e-3,
    weight_decay=0.0,
    lambda_group=1e-4,
    group_eps=1e-8,
    batch_size=128,
    max_epochs=200,
    patience=20,
    device="cpu",
):
    """
    Train DeepSurv MLP with GROUP LASSO on input layer columns (genes).

    Group lasso penalty:
        lambda_group * sum_j ||W_in[:, j]||_2
    which encourages entire gene-columns to go to (near) zero.
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(X_train, time_train, event_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_c = -np.inf
    best_epoch = -1

    for epoch in range(max_epochs):
        model.train()
        for xb, tb, eb in train_dl:
            xb = xb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)

            opt.zero_grad()
            risk = model(xb)
            npll = negative_partial_log_likelihood(risk, tb, eb)

            # Group lasso on input layer columns (genes)
            W_in = model.input_layer.weight  # shape: (hidden_dim, n_genes)
            col_l2 = torch.sqrt((W_in ** 2).sum(dim=0) + group_eps)  # (n_genes,)
            gl = lambda_group * col_l2.sum()

            loss = npll + gl
            loss.backward()
            opt.step()

        val_c, _ = eval_cindex(model, X_val, time_val, event_val, device=device)

        if val_c > best_val_c + 1e-4:
            best_val_c = val_c
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_cindex": float(best_val_c),
        "epochs_ran": int(epoch + 1),
    }

class GroupLassoCox(nn.Module):
    """
    Multi-task Cox model with group structure:

        η_i = x_i^T β_{g(i)}

    where g(i) is the group index for sample i.

    Group lasso penalty couples the β's across groups feature-wise:
        Ω_group = λ_group * sum_d ||β_{d, :}||_2
    """
    def __init__(self, input_dim: int, num_groups: int, bias: bool = True):
        super().__init__()
        # β: (num_groups, D)
        self.beta = nn.Parameter(torch.zeros(num_groups, input_dim))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_groups))
        else:
            self.bias = None

    def forward(self, x, group_idx):
        """
        x: (N, D)
        group_idx: (N,) long tensor with values in [0, num_groups-1]
        """
        # Select β for each sample
        # beta: (G, D) -> (N, D) by indexing
        beta_per_sample = self.beta[group_idx]      # (N, D)
        risk = (x * beta_per_sample).sum(dim=1)     # (N,)

        if self.bias is not None:
            b_per_sample = self.bias[group_idx]     # (N,)
            risk = risk + b_per_sample

        return torch.clamp(risk, min=-20, max=20)

    def group_lasso_penalty(self, lambda_group: float):
        """
        Group lasso over features across groups:
            sum_d ||β_{d, :}||_2
        """
        # β: (G, D) -> (D, G)
        beta_T = self.beta.t()  # (D, G)
        # L2 norm over group dimension for each feature d
        feature_group_norms = beta_T.norm(dim=1)    # (D,)
        return lambda_group * feature_group_norms.sum()

#Data-driven group lasso example

    #from sklearn.cluster import SpectralClustering
#
## affinity_sparse is your KNN graph (scipy CSR)
#n_clusters = 3  # or tune
#clustering = SpectralClustering(
#    n_clusters=n_clusters,
#    affinity="precomputed",
#    assign_labels="kmeans",
#    n_init=10,
#    random_state=0,
#)
#A_dense = affinity_sparse.toarray()
#cluster_labels = clustering.fit_predict(A_dense)  # shape (N,)

def train_epoch_group_lasso(
    model,
    optimizer,
    X_train,
    time_train,
    event_train,
    group_idx,
    batch_size=128,
    lambda_group=1e-3,
    lambda_l2=0.0,
):
    model.train()
    dataset = TensorDataset(X_train, time_train, event_train, group_idx)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for x, t, e, g in loader:
        optimizer.zero_grad()
        x = x.to(next(model.parameters()).device)
        t = t.to(x.device)
        e = e.to(x.device)
        g = g.to(x.device).long()

        risk = model(x, g)
        npll = negative_partial_log_likelihood(risk, t, e)

        loss = npll
        # Group lasso penalty
        loss = loss + model.group_lasso_penalty(lambda_group)

        # Optional extra L2 on β
        if lambda_l2 > 0:
            loss = loss + lambda_l2 * model.beta.pow(2).sum()

        loss.backward()
        optimizer.step()
    
import numpy as np
import torch
from sksurv.metrics import concordance_index_censored

def cindex_from_risk(risk_np, time_np, event_np):
    return float(concordance_index_censored(event_np.astype(bool), time_np.astype(float), risk_np)[0])

def _extract_risk_tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out

def _forward_risk_eval(model, xb, deterministic_gates: bool = True):
    """
    Evaluate risk with deterministic-gate inference when the model (or wrapped base model)
    exposes gated behavior.
    """
    if not deterministic_gates:
        return _extract_risk_tensor(model(xb))

    # Direct gated models.
    if hasattr(model, "gate_type"):
        try:
            return _extract_risk_tensor(model(xb, deterministic=True))
        except TypeError:
            return _extract_risk_tensor(model(xb))

    # Common wrappers used in this repo (RiskOnlyWrapper-like).
    for attr in ("model", "base"):
        inner = getattr(model, attr, None)
        if inner is None or not hasattr(inner, "gate_type"):
            continue
        try:
            return _extract_risk_tensor(inner(xb, deterministic=True))
        except TypeError:
            return _extract_risk_tensor(inner(xb))

    return _extract_risk_tensor(model(xb))

@torch.no_grad()
def eval_cindex(model, X, time, event, device="cpu", batch_size=512, deterministic_gates=True):
    model.eval()
    risks = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i+batch_size].to(device)
        r = _forward_risk_eval(model, xb, deterministic_gates=deterministic_gates).detach().cpu().numpy()
        risks.append(r)
    risk = np.concatenate(risks, axis=0)

    c = float(concordance_index_censored(
        event.cpu().numpy().astype(bool),
        time.cpu().numpy().astype(float),
        risk,
    )[0])

    return c, risk

def train_deepsurv_mlp_l1(
    model,
    X_train, time_train, event_train,
    X_val, time_val, event_val,
    lr=1e-3,
    weight_decay=0.0,
    lambda_l1_input=1e-4,
    batch_size=128,
    max_epochs=200,
    patience=20,
    device="cpu",
):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(X_train, time_train, event_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_state = None
    best_val_c = -np.inf
    best_epoch = -1

    for epoch in range(max_epochs):
        model.train()
        for xb, tb, eb in train_dl:
            xb = xb.to(device)
            tb = tb.to(device)
            eb = eb.to(device)

            opt.zero_grad()
            risk = model(xb)
            npll = negative_partial_log_likelihood(risk, tb, eb)

            # L1 on input layer
            W_in = model.input_layer.weight
            l1 = lambda_l1_input * W_in.abs().sum()

            loss = npll + l1
            loss.backward()
            opt.step()

        val_c, _ = eval_cindex(model, X_val, time_val, event_val, device=device)

        if val_c > best_val_c + 1e-4:
            best_val_c = val_c
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "best_val_cindex": float(best_val_c),
        "epochs_ran": int(epoch + 1),
    }

import numpy as np

def topk_genes_from_mlp(model, gene_names, K=30, norm="l2"):
    W = model.input_layer.weight.detach().cpu().numpy()
    if norm == "l1":
        scores = np.sum(np.abs(W), axis=0)
    else:
        scores = np.sqrt(np.sum(W**2, axis=0))

    idx = np.argsort(-scores)[:K]
    return scores, gene_names[idx]


# ============================
# Gated DeepSurv training utils
# ============================
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from scipy import sparse

try:
    from sksurv.metrics import concordance_index_censored
except Exception:
    concordance_index_censored = None


# ---------- helpers ----------

def _to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


def _as_float_np(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(float)
    return np.asarray(x, dtype=float)


@torch.no_grad()
def eval_cindex_and_risk(
    model: torch.nn.Module,
    X: torch.Tensor,
    time: torch.Tensor,
    event: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 2048,
    deterministic_gates: bool = True,
) -> Tuple[float, np.ndarray]:
    """
    Evaluate C-index and return (cindex, risk_scores_np).

    NOTE: requires scikit-survival available.
    """
    if concordance_index_censored is None:
        raise ImportError("scikit-survival not available: cannot compute concordance_index_censored")

    model.eval()
    X = _to_device(X, device)
    time = _to_device(time, device)
    event = _to_device(event, device)

    risks = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        risk = _forward_risk_eval(model, xb, deterministic_gates=deterministic_gates)
        risks.append(risk.detach().cpu().numpy())

    risk_np = np.concatenate(risks, axis=0).astype(float)
    c = concordance_index_censored(
        event.cpu().numpy().astype(bool),
        time.cpu().numpy().astype(float),
        risk_np
    )[0]
    return float(c), risk_np
    
def eval_patient_sparsity(
    model,
    X,
    device="cpu",
    batch_size=128,
    method="grad_x_input",
    frac_threshold=0.01,
    abs_threshold=None,
):
    """
    Returns per-patient and aggregate sparsity summaries.

    method:
      - "grad_x_input": a_ij = |x_ij * d(risk_i)/d(x_ij)|

    Metrics per patient:
      - neff_pr: participation-ratio effective genes
      - frac_active_rel: fraction of genes with a_ij >= frac_threshold * max_j a_ij
      - frac_active_abs: fraction of genes with a_ij >= abs_threshold (if provided)
    """
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    model.eval()

    ds = TensorDataset(X)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    neff_list = []
    frac_rel_list = []
    frac_abs_list = []

    for (xb,) in dl:
        xb = xb.to(device)

        if method != "grad_x_input":
            raise ValueError(f"Unknown method: {method}")

        # Need grads w.r.t. inputs
        xb_req = xb.detach().clone().requires_grad_(True)
        risk = model(xb_req)  # shape (B,) or (B,1)
        risk = risk.view(-1)

        # Compute grad of sum of risks w.r.t xb (per-sample grads come “for free”)
        grads = torch.autograd.grad(
            outputs=risk.sum(),
            inputs=xb_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        # Attribution: |x * grad|
        a = (xb_req * grads).abs()  # (B, n_genes)

        # Participation ratio effective number: (sum a)^2 / sum a^2
        a_sum = a.sum(dim=1)                      # (B,)
        a_sqsum = (a * a).sum(dim=1)              # (B,)
        neff = (a_sum * a_sum) / (a_sqsum + 1e-12)

        # Relative active fraction: >= frac_threshold * max attribution in that patient
        a_max = a.max(dim=1).values               # (B,)
        rel_cut = (frac_threshold * a_max).unsqueeze(1)
        frac_rel = (a >= rel_cut).float().mean(dim=1)

        # Absolute active fraction (optional)
        if abs_threshold is not None:
            frac_abs = (a >= abs_threshold).float().mean(dim=1)
            frac_abs_list.append(frac_abs.detach().cpu().numpy())

        neff_list.append(neff.detach().cpu().numpy())
        frac_rel_list.append(frac_rel.detach().cpu().numpy())

    neff_all = np.concatenate(neff_list, axis=0)
    frac_rel_all = np.concatenate(frac_rel_list, axis=0)
    frac_abs_all = np.concatenate(frac_abs_list, axis=0) if abs_threshold is not None else None

    out = {
        "neff_pr_per_patient": neff_all,
        "neff_pr_mean": float(neff_all.mean()),
        "neff_pr_median": float(np.median(neff_all)),
        "frac_active_rel_per_patient": frac_rel_all,
        "frac_active_rel_mean": float(frac_rel_all.mean()),
        "frac_active_rel_median": float(np.median(frac_rel_all)),
        "settings": {
            "method": method,
            "frac_threshold": frac_threshold,
            "abs_threshold": abs_threshold,
        }
    }
    if frac_abs_all is not None:
        out.update({
            "frac_active_abs_per_patient": frac_abs_all,
            "frac_active_abs_mean": float(frac_abs_all.mean()),
            "frac_active_abs_median": float(np.median(frac_abs_all)),
        })
    return out

    
@torch.no_grad()
def eval_gates_hard_K(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 2048,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, float, float]:
    """
    Hard-thresholded K computed from deterministic continuous gates
    stored in model._last_g_det_cont during forward().

    Returns:
      K_per_sample_hard: (N,) array
      K_mean, K_median
    """
    model.eval()
    X = _to_device(X, device)

    Ks = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        _risk, _gate_probs = model(xb)  # populates model._last_g_det_cont

        g_det = getattr(model, "_last_g_det_cont", None)
        if g_det is None:
            raise RuntimeError("Model did not set _last_g_det_cont in forward()")

        Kb = (g_det > threshold).sum(dim=1).detach().cpu().numpy()
        Ks.append(Kb)

    K = np.concatenate(Ks, axis=0).astype(float)
    return K, float(np.mean(K)), float(np.median(K))


@torch.no_grad()
def eval_gates_expected_K(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 2048,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      K_per_sample: (N,) array of expected number of selected features per sample
      K_mean, K_median
    """
    model.eval()
    X = _to_device(X, device)

    Ks = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        risk, gate_probs = model(xb)
        Ks.append(gate_probs.sum(dim=1).detach().cpu().numpy())

    K = np.concatenate(Ks, axis=0).astype(float)
    return K, float(np.mean(K)), float(np.median(K))


@torch.no_grad()
def gene_scores_from_gate_probs(
    model: torch.nn.Module,
    X: torch.Tensor,
    device: str = "cpu",
    batch_size: int = 2048,
    reduce: str = "mean",
) -> np.ndarray:
    """
    Aggregates gate probabilities across samples into one score per gene/feature.

    reduce:
      - "mean": mean gate prob across samples
      - "median": median gate prob across samples
    """
    model.eval()
    X = _to_device(X, device)

    gps = []
    n = X.shape[0]
    for i in range(0, n, batch_size):
        xb = X[i:i+batch_size]
        _, gate_probs = model(xb)
        gps.append(gate_probs.detach().cpu().numpy())

    G = np.concatenate(gps, axis=0)  # (N, D)

    if reduce == "mean":
        return G.mean(axis=0).astype(float)
    elif reduce == "median":
        return np.median(G, axis=0).astype(float)
    else:
        raise ValueError(f"Unknown reduce={reduce}")


def topK_genes_from_scores(scores: np.ndarray, gene_names: np.ndarray, K: int = 30):
    idx = np.argsort(-scores)[:K]
    return list(np.asarray(gene_names)[idx])


# ---------- smoothness penalties (sparse) ----------

def sample_smoothness_loss_sparse(
    gate_probs_full: torch.Tensor,
    A_sample: sparse.csr_matrix,
) -> torch.Tensor:
    """
    gate_probs_full: (N, D) tensor
    A_sample: (N, N) sparse affinity (symmetric). Nonzero entries define edges.
    Returns mean squared gate difference over edges.
    """
    coo = A_sample.tocoo()
    i = torch.tensor(coo.row, device=gate_probs_full.device, dtype=torch.long)
    j = torch.tensor(coo.col, device=gate_probs_full.device, dtype=torch.long)
    diffs = gate_probs_full[i] - gate_probs_full[j]        # (nnz, D)
    return diffs.pow(2).sum(dim=1).mean()


def gene_smoothness_loss_sparse(
    gate_probs_full: torch.Tensor,
    A_gene: sparse.csr_matrix,
) -> torch.Tensor:
    """
    gate_probs_full: (N, D)
    A_gene: (D, D) sparse affinity (symmetric). Nonzero entries define edges.

    Encourages correlated genes to have similar *usage* across samples:
      For each gene-edge (p,q), penalize ||g[:,p] - g[:,q]||^2 averaged over samples.

    Computationally: uses sparse edges and averages over samples.
    """
    coo = A_gene.tocoo()
    p = torch.tensor(coo.row, device=gate_probs_full.device, dtype=torch.long)
    q = torch.tensor(coo.col, device=gate_probs_full.device, dtype=torch.long)
    diffs = gate_probs_full[:, p] - gate_probs_full[:, q]  # (N, nnz)
    return diffs.pow(2).mean()


# ---------- main trainer ----------
from dataclasses import dataclass
from typing import Optional

@dataclass
class GatedTrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 200
    patience: int = 25
    grad_clip: float = 5.0

    # sparsity (penalizes expected K per sample)  <-- this is for module gate per patient
    lambda_sparse: float = 1e-4

    # optional: encourage a particular expected K (per sample)
    target_K: Optional[float] = None
    lambda_Kmatch: float = 0.0

    # optional smoothness penalties
    lambda_sample_smooth: float = 0.0
    lambda_gene_smooth: float = 0.0

    # compute smoothness using full-train gates once per epoch
    smooth_full_forward_every: int = 1

    # NEW: sparsity for global gene->module membership mask M (Concrete/LSPIN)
    lambda_dict_sparse: float = 0.0

    # gate aggregation (for topK stability)
    gene_score_reduce: str = "mean"



def train_gated_deepsurv(
    model: torch.nn.Module,
    X_train: torch.Tensor,
    time_train: torch.Tensor,
    event_train: torch.Tensor,
    X_val: Optional[torch.Tensor] = None,
    time_val: Optional[torch.Tensor] = None,
    event_val: Optional[torch.Tensor] = None,
    X_test: Optional[torch.Tensor] = None,
    time_test: Optional[torch.Tensor] = None,
    event_test: Optional[torch.Tensor] = None,
    *,
    config: GatedTrainConfig = GatedTrainConfig(),
    A_sample_train: Optional[sparse.csr_matrix] = None,
    A_gene: Optional[sparse.csr_matrix] = None,    
    lambda_l1_dict: float = 0.0,   # L1 on gene->module dictionary W
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Trains a gated Cox model with early stopping on validation objective (if provided).
    Validation objective matches training:
        npll + lambda_sparse * K_expected (+ lambda_Kmatch * (K_expected - target_K)^2)

    Returns dict with history + best_state etc.
    """
    model = model.to(device)

    X_train = _to_device(X_train, device)
    time_train = _to_device(time_train, device)
    event_train = _to_device(event_train, device)

    if X_val is not None:
        X_val = _to_device(X_val, device)
        time_val = _to_device(time_val, device)
        event_val = _to_device(event_val, device)
    if X_test is not None:
        X_test = _to_device(X_test, device)
        time_test = _to_device(time_test, device)
        event_test = _to_device(event_test, device)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    ds = TensorDataset(X_train, time_train, event_train)
    dl = DataLoader(ds, batch_size=config.batch_size, shuffle=True)

    # Early stopping uses val_obj if X_val provided
    best_val_obj = np.inf
    best_epoch = -1
    best_state = None

    history = {
        "train_loss": [],
        "train_Kexp_mean": [],
        "train_Khard_mean": [],
        "val_obj": [],
        "val_cindex": [],
        "val_Kexp_mean": [],
        "val_Khard_mean": [],
    }

    # Pre-check smoothness shapes if provided
    if config.lambda_sample_smooth > 0 and A_sample_train is None:
        raise ValueError("lambda_sample_smooth > 0 but A_sample_train is None")
    if config.lambda_gene_smooth > 0 and A_gene is None:
        raise ValueError("lambda_gene_smooth > 0 but A_gene is None")

    for epoch in range(config.max_epochs):
        # ============================
        # TRAIN
        # ============================
        model.train()
        losses = []
        Kexp_list = []
        Khard_list = []

        for xb, tb, eb in dl:
            if eb.sum().item() == 0:
                continue

            opt.zero_grad()
            risk, gate_probs = model(xb)

            npll = negative_partial_log_likelihood(risk, tb, eb)

            # sparsity on expected number of selected features per sample
            K_batch = gate_probs.sum(dim=1).mean()
            sparse_pen = float(config.lambda_sparse) * K_batch

            loss = npll + sparse_pen
            # NEW: global dictionary sparsity penalty (expected active gene->module memberships)
            lam_dict = float(getattr(config, "lambda_dict_sparse", 0.0))
            if lam_dict > 0.0:
                dict_gp = getattr(model, "_last_dict_gate_probs", None)
                if dict_gp is None:
                    raise RuntimeError("lambda_dict_sparse > 0 but model did not set _last_dict_gate_probs")
                # dict_gp: (G,K) probabilities
                loss = loss + lam_dict * dict_gp.mean()
            

            # optional: target-K matching penalty (expected K)
            if (config.target_K is not None) and (config.lambda_Kmatch > 0.0):
                loss = loss + float(config.lambda_Kmatch) * (K_batch - float(config.target_K)) ** 2

            loss.backward()

            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)

            opt.step()

            losses.append(float(loss.detach().item()))
            Kexp_list.append(float(K_batch.detach().item()))

            # hard K from deterministic gates (continuous) stored in forward
            g_det = getattr(model, "_last_g_det_cont", None)
            if g_det is not None:
                Kh = (g_det > 0.5).sum(dim=1).float().mean()
                Khard_list.append(float(Kh.detach().item()))
            else:
                Khard_list.append(float("nan"))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        train_Kexp_mean = float(np.mean(Kexp_list)) if Kexp_list else float("nan")
        train_Khard_mean = float(np.mean(Khard_list)) if Khard_list else float("nan")

        history["train_loss"].append(train_loss)
        history["train_Kexp_mean"].append(train_Kexp_mean)
        history["train_Khard_mean"].append(train_Khard_mean)

        # ============================
        # OPTIONAL SMOOTHNESS STEP (full forward)
        # ============================
        if (config.lambda_sample_smooth > 0 or config.lambda_gene_smooth > 0) and (
            (epoch % max(1, config.smooth_full_forward_every)) == 0
        ):
            model.train()
            opt.zero_grad()

            _, gate_full = model(X_train)  # (N_train, D)

            smooth = 0.0
            if config.lambda_sample_smooth > 0:
                s = sample_smoothness_loss_sparse(gate_full, A_sample_train)
                smooth = smooth + float(config.lambda_sample_smooth) * s

            if config.lambda_gene_smooth > 0:
                g = gene_smoothness_loss_sparse(gate_full, A_gene)
                smooth = smooth + float(config.lambda_gene_smooth) * g

            smooth.backward()
            if config.grad_clip is not None and config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()

        # ============================
        # VALIDATION + EARLY STOPPING
        # ============================
        val_c = None
        val_obj = None
        val_Kexp_mean = None
        val_Khard_mean = None

        if X_val is not None:
            # c-index for reporting
            val_c, _ = eval_cindex_and_risk(model, X_val, time_val, event_val, device=device)

            # expected + hard K summaries
            _, val_Kexp_mean, _ = eval_gates_expected_K(model, X_val, device=device)
            _, val_Khard_mean, _ = eval_gates_hard_K(model, X_val, device=device, threshold=0.5)

            # validation objective (matches training structure)
            model.eval()
            with torch.no_grad():
                risk_v, gate_probs_v = model(X_val)
                npll_v = negative_partial_log_likelihood(risk_v, time_val, event_val)
                Kexp_v = gate_probs_v.sum(dim=1).mean()

                val_obj_t = npll_v + float(config.lambda_sparse) * Kexp_v
                if (config.target_K is not None) and (config.lambda_Kmatch > 0.0):
                    val_obj_t = val_obj_t + float(config.lambda_Kmatch) * (Kexp_v - float(config.target_K)) ** 2

                val_obj = float(val_obj_t.detach().item())

            history["val_obj"].append(val_obj)
            history["val_cindex"].append(val_c)
            history["val_Kexp_mean"].append(float(val_Kexp_mean))
            history["val_Khard_mean"].append(float(val_Khard_mean))

            # early stopping on val_obj (minimize)
            improved = (val_obj < best_val_obj - 1e-6)
            if improved:
                best_val_obj = val_obj
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            elif (epoch - best_epoch) >= config.patience and best_epoch >= 0:
                if verbose:
                    print(f"[early stop] epoch={epoch} best_epoch={best_epoch} best_val_obj={best_val_obj:.6f}")
                break

        # ============================
        # PRINT
        # ============================
        if verbose:
            msg = (
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| Kexp_train~{train_Kexp_mean:.1f} | Khard_train~{train_Khard_mean:.1f}"
            )
            if val_c is not None:
                msg += (
                    f" | val_obj={val_obj:.4f} | val_c={val_c:.4f} "
                    f"| Kexp_val~{val_Kexp_mean:.1f} | Khard_val~{val_Khard_mean:.1f}"
                )
            print(msg)

    # restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    out: Dict[str, Any] = {
        "best_epoch": int(best_epoch),
        "best_val_obj": float(best_val_obj) if best_epoch >= 0 else None,
        "best_val_cindex": float(np.max(history["val_cindex"])) if len(history["val_cindex"]) else None,
        "history": history,
        "state_dict_best": best_state,
        "model": model,
    }

    # train metrics (on full train)
    try:
        c_tr, _ = eval_cindex_and_risk(model, X_train, time_train, event_train, device=device)
        out["train_cindex"] = float(c_tr)
    except Exception:
        out["train_cindex"] = None

    # expected and hard K on train
    _, K_tr_mean, K_tr_med = eval_gates_expected_K(model, X_train, device=device)
    _, Kh_tr_mean, Kh_tr_med = eval_gates_hard_K(model, X_train, device=device, threshold=0.5)
    out["Kexp_train_mean"] = float(K_tr_mean)
    out["Kexp_train_median"] = float(K_tr_med)
    out["Khard_train_mean"] = float(Kh_tr_mean)
    out["Khard_train_median"] = float(Kh_tr_med)

    # gene scores on train (still based on gate_probs, i.e. expected usage)
    out["gene_scores_train"] = gene_scores_from_gate_probs(
        model, X_train, device=device, reduce=config.gene_score_reduce
    )

    # test metrics
    if X_test is not None:
        c_te, _ = eval_cindex_and_risk(model, X_test, time_test, event_test, device=device)
        out["test_cindex"] = float(c_te)

        _, K_te_mean, K_te_med = eval_gates_expected_K(model, X_test, device=device)
        _, Kh_te_mean, Kh_te_med = eval_gates_hard_K(model, X_test, device=device, threshold=0.5)
        out["Kexp_test_mean"] = float(K_te_mean)
        out["Kexp_test_median"] = float(K_te_med)
        out["Khard_test_mean"] = float(Kh_te_mean)
        out["Khard_test_median"] = float(Kh_te_med)

        out["gene_scores_test"] = gene_scores_from_gate_probs(
            model, X_test, device=device, reduce=config.gene_score_reduce
        )
    else:
        out["test_cindex"] = None

    return out



# ---------- convenience: lambda sweep to hit target K ----------

def sweep_lambda_sparse_for_targetK(
    make_model_fn: Callable[[], torch.nn.Module],
    X_train, time_train, event_train,
    X_val, time_val, event_val,
    *,
    lambdas=(1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3),
    targetK: float = 30.0,
    config_base: Optional[GatedTrainConfig] = None,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Fits short runs over lambda_sparse values and returns the run closest to targetK (by K_val_mean).
    """
    if config_base is None:
        config_base = GatedTrainConfig(max_epochs=120, patience=15)

    records = []
    best = None
    best_score = np.inf

    for lam in lambdas:
        cfg = GatedTrainConfig(**{**config_base.__dict__})
        cfg.lambda_sparse = float(lam)

        m = make_model_fn()
        res = train_gated_deepsurv(
            m,
            X_train, time_train, event_train,
            X_val, time_val, event_val,
            config=cfg,
            device=device,
            verbose=False,
        )
        K_val_mean = float(res["history"]["val_K_mean"][-1]) if len(res["history"]["val_K_mean"]) else float(res["K_train_mean"])
        val_c = res["best_val_cindex"]

        records.append({
            "lambda_sparse": lam,
            "best_val_cindex": val_c,
            "K_val_mean": K_val_mean,
            "best_epoch": res["best_epoch"],
        })

        score = abs(K_val_mean - targetK)
        if score < best_score:
            best_score = score
            best = res

        if verbose:
            print(f"λ={lam:.2e}  K_val~{K_val_mean:.1f}  val_c={val_c}")

    return {
        "records": records,
        "best": best,
        "best_targetK_error": float(best_score),
    }

import torch
import numpy as np

@torch.no_grad()
def gate_det_lspin_tf(model, X, device="cpu"):
    """
    Deterministic TF-style gate probabilities:
        g_det = clip(a * alpha + 0.5, 0, 1)
    where alpha = gating_net(x).

    Returns: (N, D) tensor on CPU
    """
    model.eval()
    X = X.to(device)
    alpha = model.gating_net(X)
    g_det = torch.clamp(model.a * alpha + 0.5, 0.0, 1.0)
    return g_det.detach().cpu()

@torch.no_grad()
def expected_K_lspin_tf(model, X, device="cpu"):
    """
    Returns:
      K_mean: float
      K_per_sample: numpy array (N,)
    """
    g_det = gate_det_lspin_tf(model, X, device=device)   # (N, D)
    K_per = g_det.sum(dim=1).numpy()
    return float(K_per.mean()), K_per
@torch.no_grad()
def alpha_and_gdet(self, x):
    alpha = self.gating_net(x)
    g_det = torch.clamp(self.a * alpha + 0.5, 0.0, 1.0)
    return alpha, g_det
