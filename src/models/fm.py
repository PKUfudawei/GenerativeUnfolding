import torch, torchdiffeq
import torch.nn as nn
from .SiT import SiT
from ..train.utils import mmd2_loss


class FlowMatching(nn.Module):
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
        self.net = SiT(depth=4, hidden_size=64, num_heads=2, mlp_ratio=2.0, use_conditions=False)

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


    def batch_loss(self, target, source=None, kl_scale=None, cond=None):
        data, noise = target, source

        if noise is None:
            noise = torch.randn_like(data)

        # Sample time steps
        t = torch.rand(data.shape[0], device=data.device)
        const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))
        t_ = t.reshape(const_shape).detach().clone()
        z_t = (1 - t_) * data + t_ * noise

        v_t = self.net(x=z_t, t=t, r=t, cond=cond, y=None)

        error = v_t - (noise - data)
        loss = self.adaptive_l2_loss(error, gamma=1)
        loss_mean_ref = (error.detach() ** 2).mean()
        with torch.no_grad():
            z0_theta = self.sample(source=source, cond=cond)
        loss_mmd = mmd2_loss(z0_theta, data, sigma=1)
        loss_terms = {
            "loss": loss.item(),
            "loss_mean_ref": loss_mean_ref.item(),
            "loss_mmd": loss_mmd.item()
        }

        return loss, loss_terms
    
    def sample(self, source, num_steps=20, cond=None):
        if source is None:
            source = torch.randn_like(cond)
        batch_size = source.shape[0]
        
        x0_sample = source
        time_schedule = torch.linspace(1.0, 0.0, num_steps + 1, device=source.device)
        for t in torch.arange(0, num_steps, device=source.device):
            time = time_schedule[t]
            x0_sample -= (time_schedule[t+1] - time) * self.net(x=x0_sample, t=time.repeat(batch_size), r=time.repeat(batch_size), cond=cond, y=None)

        return x0_sample
