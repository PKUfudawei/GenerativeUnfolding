
def stopgrad(x):
    return x.detach()

def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def _layer_sums(x, layer_dim=2):
    """
    x: (B, C, L, H, W) or (B, L, H, W)
    returns: (B, L)
    """
    if x.ndim == 5:   # (B, C, L, H, W)
        return x.sum(dim=(1, 3, 4)).squeeze(1)  # sum over C,H,W -> (B,L)
    elif x.ndim == 4: # (B, L, H, W)
        return x.sum(dim=(2, 3))                # sum over H,W   -> (B,L)
    else:
        raise ValueError(f"Unexpected shape for layer_sums: {x.shape}")


def adaptive_l2_loss(error, gamma=0.5, c=1e-3):
    """
    Adaptive L2 loss: sg(w) * ||Δ||_2^2, where w = 1 / (||Δ||^2 + c)^p, p = 1 - γ
    Args:
        error: Tensor of shape (B, C, W, H)
        gamma: Power used in original ||Δ||^{2γ} loss
        c: Small constant for stability
    Returns:
        Scalar loss
    """
    if error.ndim == 4:
        # old: (B, C, H, W)
        delta_sq = torch.mean(error**2, dim=(1,2,3))
    elif error.ndim == 5:
        delta_sq = torch.mean(error**2, dim=(1, 2, 3, 4))
    elif error.ndim == 2:
        # new: (B, D)
        delta_sq = torch.mean(error**2, dim=1)
    p = 1.0 - gamma
    w = 1.0 / (delta_sq + c).pow(p)
    loss = delta_sq  # ||Δ||^2
    return (stopgrad(w) * loss).mean()  


# fix: r should be always not larger than t
def sample_t_r(self, batch_size, device, flow_ratio=0.75):
    if self.time_dist[0] == 'uniform':
        samples = np.random.rand(batch_size, 2).astype(np.float32)

    elif self.time_dist[0] == 'lognorm':
        mu, sigma = self.time_dist[-2], self.time_dist[-1]
        normal_samples = np.random.randn(batch_size, 2).astype(np.float32) * sigma + mu
        samples = 1 / (1 + np.exp(-normal_samples))  # Apply sigmoid

    # Assign t = max, r = min, for each pair
    t_np = np.maximum(samples[:, 0], samples[:, 1])
    r_np = np.minimum(samples[:, 0], samples[:, 1])

    num_selected = int(flow_ratio * batch_size)
    indices = np.random.permutation(batch_size)[:num_selected]
    r_np[indices] = t_np[indices]

    t = torch.tensor(t_np, device=device)
    r = torch.tensor(r_np, device=device)
    return t, r


def compute_loss_meanflow(self, data, energy, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
    self.scales = scales
    device = data.device
    dtype  = data.dtype
    
    time_dist=['lognorm', -0.4, 1.0]
    self.time_dist = time_dist

    if noise is None:
        noise = torch.randn_like(data)

    # sample t
    batch_size = data.shape[0]
    const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

        
    # Sample time steps
    #r, t = self.sample_time_steps(batch_size, device, time_sampler)
    r, t = self.sample_t_r(batch_size, device, 0.75)
    
    t_ = torch.reshape(t, const_shape).detach().clone()
    r_ = torch.reshape(r, const_shape).detach().clone()

    # interpolant
    #alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(t.view(const_shape))

    # model IO
    #z_t = alpha_t * data + sigma_t * noise
    #v_t = d_alpha_t * data + d_sigma_t * noise
    
    z_t = (1 - t_) * data + t_ * noise
    v_t = noise - data
    
    time_diff = (t - r).view(const_shape)
            
    u_target = torch.zeros_like(v_t)
    
    energy = energy.to(dtype=dtype)
    
    
    u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
    
    primals = (z_t, r, t)
    tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
    
    def fn_current(z, cur_r, cur_t):
        return self.pred_meanflow(z, energy, cur_t, r_emb = cur_r, layers=layers)
    
    #_, dudt = jvp(fn_current,primals,tangents)
    
    u, dudt = jvp(fn_current,primals,tangents)
    
    
    u_target = v_t - time_diff * dudt

    # Detach the target to prevent gradient flow        
    #error = u - u_target.detach()
    

    
    error = u - stopgrad(u_target)
    loss_mid = adaptive_l2_loss(error)
    # loss = F.mse_loss(u, stopgrad(u_tgt))

    loss_mean_ref = (stopgrad(error) ** 2).mean()
    #loss_mid = torch.sum((error**2).reshape(error.shape[0],-1), dim=-1)
    
    # Apply adative weighting based on configuration
    
    
    ### adding this for removing large loss for many voxels
    #numel_per = error[0].numel()
    #loss_mid = loss_mid / numel_per
    
    
    if adaptive:
        weights = 1.0 / (loss_mid.detach() + 1e-3).pow(1)
        loss = weights * loss_mid          
    else:
        loss = loss_mid
    #loss_mean_ref = torch.mean((error**2))
    
        
    return loss, loss_mean_ref



def compute_loss_imeanflow(self, data, energy, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
    self.scales = scales
    device = data.device
    dtype  = data.dtype
    
    time_dist=['lognorm', -0.4, 1.0]
    self.time_dist = time_dist

    if noise is None:
        noise = torch.randn_like(data)

    # sample t
    batch_size = data.shape[0]
    const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

        
    # Sample time steps
    #r, t = self.sample_time_steps(batch_size, device, time_sampler)
    r, t = self.sample_t_r(batch_size, device, 0.75)
    
    t_ = torch.reshape(t, const_shape).detach().clone()
    r_ = torch.reshape(r, const_shape).detach().clone()

    
    z_t = (1 - t_) * data + t_ * noise
    #v_t = noise - data
    v_t = self.pred_meanflow(z_t, energy, t, t, layers=layers)
    
    time_diff = (t - r).view(const_shape)
            
    u_target = torch.zeros_like(v_t)
    
    energy = energy.to(dtype=dtype)
    
    
    #u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
    
    primals = (z_t, r, t)
    tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
    
    def fn_current(z, cur_r, cur_t):
        return self.pred_meanflow(z, energy, cur_t, r_emb = cur_r, layers=layers)
    
    #_, dudt = jvp(fn_current,primals,tangents)
    
    u, dudt = jvp(fn_current,primals,tangents)
    
    
    V = u + time_diff * stopgrad(dudt)

    # Detach the target to prevent gradient flow        
    #error = u - u_target.detach()
    

    
    error = V - (noise - data)
    loss_mid = adaptive_l2_loss(error)
    # loss = F.mse_loss(u, stopgrad(u_tgt))

    loss_mean_ref = (stopgrad(error) ** 2).mean()
    
    
    if adaptive:
        weights = 1.0 / (loss_mid.detach() + 1e-3).pow(1)
        loss = weights * loss_mid          
    else:
        loss = loss_mid
    #loss_mean_ref = torch.mean((error**2))
    
        
    return loss, loss_mean_ref

### experimental loss function for inductive momentum matching

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
    D = np.prod(x.shape[1:])
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


def compute_loss_meanflow_imm_v1(self, data, energy, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
    self.scales = scales
    device = data.device
    dtype  = data.dtype
    
    time_dist=['lognorm', -0.4, 1.0]
    self.time_dist = time_dist

    if noise is None:
        noise = torch.randn_like(data)

    # sample t
    batch_size = data.shape[0]
    const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

    r, t = self.sample_t_r(batch_size, device, 0.75)
    
    t_ = torch.reshape(t, const_shape).detach().clone()
    r_ = torch.reshape(r, const_shape).detach().clone()

    
    z_t = (1 - t_) * data + t_ * noise
    v_t = noise - data
    
    time_diff = (t - r).view(const_shape)
            
    u_target = torch.zeros_like(v_t)
    
    energy = energy.to(dtype=dtype)
    
    
    u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
    
    primals = (z_t, r, t)
    tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
    
    def fn_current(z, cur_r, cur_t):
        return self.pred_meanflow(z, energy, cur_t, r_emb = cur_r, layers=layers)
    
    
    u, dudt = jvp(fn_current,primals,tangents)
    
    
    u_target = v_t - time_diff * dudt
    
    z0_theta = self.sample_one_step(noise, energy, layers=layers)
    loss_mmd = mmd2_loss(z0_theta, data, sigma=1)



    error = u - stopgrad(u_target)
    loss_mid = adaptive_l2_loss(error)

    loss_mean_ref = (stopgrad(error) ** 2).mean()
    
    
    if adaptive:
        weights = 1.0 / (loss_mid.detach() + 1e-3).pow(1)
        loss = weights * loss_mid          
    else:
        loss = loss_mid
        
    return loss, loss_mean_ref, loss_mmd


def compute_loss_meanflow_imm_v2(self, data, energy, noise = None, t = None, layers = None, weighting = 'uniform', energy_loss_scale = 1e-2, scales=1, time_sampler="uniform", time_mu=-0.4,time_sigma=1.0, adaptive=False):
    self.scales = scales
    device = data.device
    dtype  = data.dtype
    
    time_dist=['lognorm', -0.4, 1.0]
    self.time_dist = time_dist

    if noise is None:
        noise = torch.randn_like(data)

    # sample t
    batch_size = data.shape[0]
    const_shape = (data.shape[0], *((1,) * (len(data.shape) - 1)))

    r, t = self.sample_t_r(batch_size, device, 0.75)
    
    t_ = torch.reshape(t, const_shape).detach().clone()
    r_ = torch.reshape(r, const_shape).detach().clone()

    
    z_t = (1 - t_) * data + t_ * noise
    v_t = noise - data
    
    time_diff = (t - r).view(const_shape)
            
    u_target = torch.zeros_like(v_t)
    
    energy = energy.to(dtype=dtype)
    
    
    u = self.pred_meanflow(z_t, energy, t,r_emb = r, layers=layers)
    
    primals = (z_t, r, t)
    tangents = (v_t, torch.zeros_like(r), torch.ones_like(t))
    
    def fn_current(z, cur_r, cur_t):
        return self.pred_meanflow(z, energy, cur_t, r_emb = cur_r, layers=layers)
    
    
    u, dudt = jvp(fn_current,primals,tangents)
    
    
    u_target = v_t - time_diff * dudt
    
    z0_theta = self.sample_one_step(noise, energy, layers=layers)
    loss_mmd = mmd2_loss(z0_theta, data, sigma=1)



    error = u - stopgrad(u_target)
    loss_mid = adaptive_l2_loss(error)

    loss_mean_ref = (stopgrad(error) ** 2).mean()
    
    
    if adaptive:
        weights = 1.0 / (loss_mid.detach() + 1e-3).pow(1)
        loss = weights * loss_mid          
    else:
        loss = loss_mid
        
    return loss, loss_mean_ref, loss_mmd