import torch
from torch.func import jvp
import torch.nn as nn
import numpy as np
from .SiT import SiT


def mmd2_kernel(x, y, sigma=1.0, eps=1e-8):
    """
    Your kernel style, but without time axis and without w.
    x: [B, ...], y: [B, ...]
    Returns: [B, B] kernel matrix.
    """
    # x -> [B, 1, ...], y -> [1, B, ...]
    x_exp = x.unsqueeze(1)
    y_exp = y.unsqueeze(0)

    # flatten over all non-batch dims
    flatten_dim = 2
    diff2 = ((x_exp - y_exp) ** 2).flatten(flatten_dim).sum(-1)  # [B, B]

    # L2 / (D * sigma)
    D = torch.prod(torch.tensor(x.shape[1:]))
    dist = torch.clamp_min(diff2, eps).sqrt() / (D * sigma)

    K = torch.exp(-dist)  # w = 1
    return K


def mmd2_loss(x, y, sigma=1.0):
    """
    MMD^2 using the above kernel.
    x, y: [B, ...]
    """
    Kxx = mmd2_kernel(x, x, sigma=sigma)
    Kyy = mmd2_kernel(y, y, sigma=sigma)
    Kxy = mmd2_kernel(x, y, sigma=sigma)

    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


class MeanFlow(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.bayesian = params.get("bayesian", False)
        self.bayesian_samples = params.get("bayesian_samples", 20)
        self.bayesian_layers = []
        self.bayesian_factor = params.get("bayesian_factor", 1)
        self.reco_jets = self.params["process_params"].get("reco_jets", 4)
        self.build_net()
        print(f"    Using reco_jets {self.reco_jets}")

    def build_net(self):
        self.net = SiT(depth=6, hidden_size=128, num_heads=4, mlp_ratio=2.0, use_conditions=False)

    def adaptive_l2_loss(self, error, gamma=0.5, c=1e-3):
        """
        Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
        Args:
            error: Tensor of shape (B, C, W, H)
            gamma: Power used in original ||Δ||^{2γ} loss
            c: Small constant for stability
        Returns:
            Scalar loss
        """
        loss_per_sample = torch.mean(error**2, dim=tuple(range(1, error.ndim)), keepdim=False)
        p = 1.0 - gamma
        weight = 1.0 / (loss_per_sample + c).pow(p)
        return (weight.detach() * loss_per_sample).mean()  


    def sample_t_r(self, batch_size, flow_ratio=0.75):
        if self.time_dist[0] == 'uniform':
            samples = torch.rand(batch_size, 2)

        elif self.time_dist[0] == 'lognorm':
            mu, sigma = self.time_dist[-2], self.time_dist[-1]
            normal_samples = torch.randn(batch_size, 2) * sigma + mu
            samples = 1 / (1 + torch.exp(-normal_samples))  # Apply sigmoid

        t = samples.max(dim=1).values
        r = samples.min(dim=1).values

        num_selected = int(flow_ratio * batch_size)
        indices = torch.randperm(batch_size)[:num_selected]
        r[indices] = t[indices]

        return t, r


    def batch_loss(self, target, source=None, kl_scale=None, cond=None):
        data, noise = target, source
        time_dist=['lognorm', -0.4, 1.0]
        self.time_dist = time_dist

        if noise is None:
            noise = torch.randn_like(data)

        # Sample time steps
        t, r = self.sample_t_r(data.shape[0], 0.75)
        t, r = t.to(device=data.device), r.to(device=data.device)
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        t_ = t.reshape(const_shape).detach().clone()
        z_t = (1 - t_) * data + t_ * noise

        if self.params.get('iMF', True):
            v_t = self.net(x=z_t, t=t, r=t, cond=cond, y=None)
        else:
            v_t = noise - data

        u, dudt = jvp(
            lambda z, t, r: self.net(x=z, t=t, r=r, cond=cond, y=None),
            (z_t, t, r), 
            (v_t, torch.ones_like(t), torch.zeros_like(r))
        )
        u_target = noise - data - (t - r).view(const_shape) * dudt

        error = u - u_target.detach()
        loss = self.adaptive_l2_loss(error, gamma=1)
        loss_mean_ref = (error.detach() ** 2).mean()
        z0_theta = self.sample(noise=noise, cond=cond)
        loss_mmd = mmd2_loss(z0_theta, data, sigma=1)
        loss_terms = {
            "loss": loss.item(),
            "loss_mean_ref": loss_mean_ref.item(),
            "loss_mmd": loss_mmd.item()
        }

        return loss, loss_terms
    
    def sample(self, source, cond=None):
        if source is None:
            source = torch.randn_like(cond)
        batch_size = source.shape[0]
        t = torch.ones(batch_size, device=source.device)
        r = torch.zeros(batch_size, device=source.device)

        u_sample = self.net(x=source, t=t, r=r, cond=cond, y=None)
        x0_sample = source - u_sample

        return x0_sample