from typing import Tuple, Optional, Union, Iterable
import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
from math import pi, gamma
from .utils import *
import scipy

# Define Metric
MINKOWSKI = torch.diag(torch.tensor([1.0, -1.0, -1.0, -1.0]))


class PreprocTrafo(nn.Module):
    """
    Base class for a preprocessing transformation. It allows for different input and
    output shapes and both non-invertible as well as invertible transformations with or
    without known Jacobian
    """

    def __init__(
        self,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        invertible: bool
    ):
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.invertible = invertible

    def forward(
        self,
        x: torch.Tensor,
        rev: bool = False,
        batch_size: int = 100000,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        batch_size = len(x)
        if rev and not self.invertible:
            raise ValueError("Tried to call inverse of non-invertible transformation")
        input_shape, output_shape = (
            (self.output_shape, self.input_shape)
            if rev
            else (self.input_shape, self.output_shape)
        )
        if x.shape[1:] != input_shape:
            raise ValueError(
                f"Wrong input shape. Expected {input_shape}, "
                + f"got {tuple(x.shape[1:])}"
            )

        ybs = []
        for xb in x.split(batch_size, dim=0):
            yb = self.transform(xb, rev)
            ybs.append(yb)
        y = torch.cat(ybs, dim=0)

        if y.shape[1:] != output_shape:
            raise ValueError(
                f"Wrong output shape. Expected {output_shape}, "
                + f"got {tuple(y.shape[1:])}"
            )
        return y


class PreprocChain(PreprocTrafo):
    def __init__(
        self,
        trafos: Iterable[PreprocTrafo],
        normalize: bool = True
    ):
        if any(
            tp.output_shape != tn.input_shape
            for i, (tp, tn) in enumerate(zip(trafos[:-1], trafos[1:]))
        ):
            raise ValueError(
                f"Output shape {trafos[0].output_shape} of transformation {0} not "
                + f"equal to input shape {trafos[1].input_shape} of transformation {1}"
            )

        trafos.append(NormalizationPreproc(trafos[-1].output_shape))
        super().__init__(
            trafos[0].input_shape,
            trafos[-1].output_shape,
            all(t.invertible for t in trafos)
        )
        self.trafos = nn.ModuleList(trafos)
        self.normalize = normalize

    def init_normalization(self, x: torch.Tensor, batch_size: int = 100000):
        if not self.normalize:
            return
        xbs = []
        for xb in x.split(batch_size, dim=0):
            for t in self.trafos[:-1]:
                xb = t(xb)
            xbs.append(xb)
        x = torch.cat(xbs, dim=0)
        norm_dims = tuple(range(len(x.shape)-1))
        x_mean = np.nanmean(x.clone().detach().cpu().numpy(), axis=norm_dims, keepdims=True)
        x_std = np.nanstd(x.clone().detach().cpu().numpy(), axis=norm_dims, keepdims=True)
        #x_mean = x.mean(dim=norm_dims, keepdims=True)
        #x_std = x.std(dim=norm_dims, keepdims=True)
        self.mean = torch.tensor(x_mean, dtype=x.dtype, device=x.device)
        self.std = torch.tensor(x_std, dtype=x.dtype, device=x.device)
        self.trafos[-1].set_norm(self.mean[0].expand(x.shape[1:]), self.std[0].expand(x.shape[1:]))

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        for t in reversed(self.trafos) if rev else self.trafos:
            x = t(x, rev=rev)
        return x


class NormalizationPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...]
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.mean = nn.Parameter(torch.zeros(shape), requires_grad=False)
        self.std = nn.Parameter(torch.ones(shape), requires_grad=False)

    def set_norm(self, mean: torch.Tensor, std: torch.Tensor):
        with torch.no_grad():
            self.mean.data.copy_(mean)
            self.std.data.copy_(std)

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x * self.std + self.mean
        else:
            z = (x - self.mean) / self.std
        return z


class UniformNoisePreprocessing(PreprocTrafo):
    def __init__(self, shape: Tuple[int, ...], channels):

        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x.clone()
            z[:, self.channels] = torch.round(z[:, self.channels])
        else:
            z = x.clone()
            noise = torch.rand_like(z[:, self.channels])-0.5
            z[:, self.channels] = z[:, self.channels] + noise
        return z


class ErfPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...], channels
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x
            z[:, self.channels] = torch.erf(z[:, self.channels]) + .001
        else:
            z = x
            z[:, self.channels] = torch.erfinv(z[:, self.channels] - .001)
        return z


class CubicRootPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...], channels
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x.clone()
            z[:, self.channels] = z[:, self.channels] ** 3
        else:
            z = x.clone()
            z[:, self.channels] = np.cbrt(z[:, self.channels])# ** (1./3.)
        return z


class LogPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...], channels
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)
        self.channels = channels

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x.clone()
            z[:, self.channels] = z[:, self.channels].exp()
            if 3 in self.channels:
                z[:, 3] = -1 * z[:, 3]
        else:
            z = x.clone()
            if 3 in self.channels:
                z[:, 3] = -1 * z[:, 3]
            z[:, self.channels] = (z[:, self.channels]).log()
        return z


class SpecialPreproc(PreprocTrafo):
    def __init__(
        self,
        shape: Tuple[int, ...]
    ):
        super().__init__(input_shape=shape, output_shape=shape, invertible=True)

    def transform(self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            z = x.clone()
            z4 = z[:, 4]
            z4 = torch.erf(z4)
            z4 = z4*self.factor
            z4 = z4+self.shift
            z4 = z4.exp()
            z4 = torch.where(z4 < 0.1, 0, z4)
            z[:, 4] = z4
        else:
            z = x.clone()
            z4 = z[:, 4]
            noise = torch.rand(size=z4.shape, device=x.device)/1000. * 3 + 0.097
            z4 = torch.where(z4 < 0.1, noise, z4)
            z4 = z4.log()
            self.shift = (z4.max() + z4.min())/2.
            z4 = z4-self.shift
            self.factor = max(z4.max(), -1 * z4.min())*1.001
            z4 = z4/self.factor
            z4 = torch.erfinv(z4)
            z[:, 4] = z4
        return z


def build_preprocessing(params: dict):
    """
    Builds a preprocessing chain with the given parameters

    Args:
        params: dictionary with preprocessing parameters
    Returns:
        Preprocessing chain
    """
    type = params.get("type", "ZJets")
    print(f"    Preprocessing: {params}")
    if type == "ZJets":
        print(f"    Preprocessing: ZJets")
        n_dim = 6
        normalize = True
        uniform_noise_channels = params.get("uniform_noise_channels", [])
        cubic_root_channels = params.get("cubic_root_channels", [])
        log_channels = params.get("log_channels", [])
        special_preproc = params.get("special_preproc", False)

        trafos = []
        if len(uniform_noise_channels) != 0:
            trafos.append(UniformNoisePreprocessing(shape=(n_dim,), channels=uniform_noise_channels))
        if len(cubic_root_channels) != 0:
            trafos.append(CubicRootPreproc(shape=(n_dim,), channels=cubic_root_channels))

        if len(log_channels) != 0:
            trafos.append(LogPreproc(shape=(n_dim,), channels=log_channels))

        if special_preproc:
            trafos.append(SpecialPreproc(shape=(n_dim,)))

        if len(trafos) == 0:
            class IdentityPreproc(PreprocTrafo):
                def __init__(self, shape):
                    super().__init__(input_shape=shape, output_shape=shape, invertible=True)
                def transform(self, x, rev=False):
                    return x
            trafos.append(IdentityPreproc(shape=(n_dim,)))

        return PreprocChain(
            trafos,
            normalize=normalize
        )

    elif type == "ttbar_naive_hard":

        trafos = [jet_coordinates_hard(log=params.get("log", False),
                                       drop_masses=params.get("drop_masses", False),
                                       erf_phi=params.get("erf_phi", False),
                                       hadronic_first=params.get("hadronic_first", False))]
        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )

    elif type == "ttbar_naive_reco":
        old_reco = params.get("old_reco", False)
        trafos = [jet_coordinates_reco(log=params.get("log", False),
                                       reco_jets=params.get("reco_jets", 4),
                                       btag=params.get("btag", False),
                                       old_reco=old_reco),
                                        ]
        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )

    elif type == "ttbar_cartesian_reco":
        trafos = [cartesian_reco(log=params.get("log", False),
                                       reco_jets=params.get("reco_jets", 4))]
        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )

    elif type == "ttbar_massparam":
        drop_masses = params.get("drop_masses", False)
        old_massparam = params.get("old_massparam", False)
        #print(old_massparam)
        if old_massparam:
            print("Using massparam")
            trafos = [ttbar_massparam(breit_wigner=params.get("breit_wigner", False))]
        else:
            print("Using massparam_v2")
            trafos = [ttbar_massparam_v2(breit_wigner=params.get("breit_wigner", False),
                                         mW_first=params.get("mW_first", False))]

        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )
    
    elif type == "ttbar_cartesian_hard":
        trafos = [cartesian_hard(mppp=params.get("mppp", True),
                                 drop_masses=params.get("drop_masses", False),
                                 hadronic_first=params.get("hadronic_first", False))]
        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )
    elif type == "transfermer_reco":
        trafos = [transfermer_reco(log=params.get("log", True), eta_cut=params.get("eta", False))]
        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )

    elif type == "transfermer_hard":
        trafos = [transfermer_hard(log=params.get("log", True), eta_cut=params.get("eta", False))]
        normalize = True
        return PreprocChain(
            trafos,
            normalize=normalize
        )


class jet_coordinates_reco(PreprocTrafo):
    def __init__(
        self,
        reco_jets=4,
        log=True,
        btag=False,
        old_reco=False
    ):
        #reco_jets = 4
        n_particles = 2+reco_jets
        output_dim = 3+4+reco_jets*4 + int(btag)*reco_jets - int(old_reco)
        self.reco_jets = reco_jets
        self.log = log
        self.btag = btag
        self.old_reco = old_reco
        super().__init__(
            input_shape=(n_particles, 5),
            output_shape=(output_dim,),
            invertible=True
        )

    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.naive_reverse(x)
        else:
            return self.naive_forward(x)

    def naive_forward(self, p):
        x = p.clone()
        if self.log:
            x[:, :2, 2] = (x[:, :2, 2] + 1.e-4).log()
            x[:, 2:, 1:3] = (x[:, 2:, 1:3] + 1.e-4).log()
        if not self.btag:
            x = x[:, :, 1:]
            x = x.reshape(x.shape[0], -1)
            keep_channels = [i for i in np.arange(1, x.shape[1])]
            if self.old_reco:
                keep_channels.remove(2)
        else:
            x = x.reshape(x.shape[0], -1)
            keep_channels = [i for i in np.arange(2, x.shape[1]) if i not in [5]]
        x = x[:, keep_channels]
        return x


class cartesian_reco(PreprocTrafo):
    def __init__(
        self,
        reco_jets=4,
        log=True
    ):
        reco_jets = 4
        n_particles = 2+reco_jets
        output_dim = 3+4+reco_jets*4
        self.max_reco_jets = reco_jets
        self.log = log
        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(output_dim,),
            invertible=True
        )

    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.naive_reverse(x)
        else:
            return self.naive_forward(x)

    def naive_forward(self, p):
        y = p.clone().detach().cpu().numpy()
        x = jet2cartesian(y)
        x[:, :, 0] = y[:, :, 0]
        if self.log:
            x[:, 2:, 0] = np.log(x[:, 2:, 0] + 1.e-4)
        x = x.reshape(x.shape[0], -1)
        keep_channels = [i for i in np.arange(1, 4+4+4*self.max_reco_jets)]
        x = x[:, keep_channels]
        return torch.tensor(x, dtype=p.dtype, device=p.device)


class jet_coordinates_hard(PreprocTrafo):
    def __init__(
        self,
        log=False,
        drop_masses=False,
        erf_phi=False,
        hadronic_first=False
    ):
        self.log = log
        self.drop_masses = drop_masses
        self.erf_phi = erf_phi
        self.hadronic_first = hadronic_first

        n_particles = 10
        output_dim = 20 - int(drop_masses) #+ 1

        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(output_dim,),
            invertible=True
        )

        self.m_b = 4.7

    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.naive_reverse(x)
        else:
            return self.naive_forward(x)

    def naive_forward(self, p):
        if self.hadronic_first:
            x = p.clone()[:, [6, 8, 9, 1, 3, 4]]
        else:
            x = p.clone()[:, [1, 3, 4, 6, 8, 9]]
        if self.log:
            x[:, :, 1] = (x[:, :, 1]+1.e-4).log()
        if self.erf_phi:
            x[:, :, 3] = torch.erfinv(0.999 * x[:, :, 3]/np.pi)
        x = x.reshape(x.shape[0], -1)
        # kick b mass, neutrino mass, bqbar mass
        if not self.drop_masses:
            keep_channels = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]
        else:
            keep_channels = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
            keep_channels = keep_channels + [4] if self.hadronic_first else keep_channels + [16]
        x = x[:, keep_channels]
        return x

    def naive_reverse(self, p):
        y = p.clone().detach().cpu()
        x = np.zeros((p.shape[0], 10, 4))
        if not self.drop_masses:
            x[:, 1, 0] = self.m_b
            x[:, 1, 1:] = y[:, :3]
            x[:, 3, :] = y[:, 3:7]
            x[:, 4, 1:] = y[:, 7:10]
            x[:, 6, 0] = self.m_b
            x[:, 6, 1:] = y[:, 10:13]
            x[:, 8, :] = y[:, 13:17]
            x[:, 9, 1:] = y[:, 17:20]
        else:
            x[:, 1, 0] = self.m_b
            x[:, 1, 1:] = y[:, :3]
            x[:, 3, 1:] = y[:, 3:6]
            x[:, 4, 1:] = y[:, 6:9]
            x[:, 6, 0] = self.m_b
            x[:, 6, 1:] = y[:, 9:12]
            x[:, 8, 1:] = y[:, 12:15]
            x[:, 9, 1:] = y[:, 15:18]

            if self.hadronic_first:
                x[:, 3, 0] = y[:, -1]
            else:
                x[:, 8, 0] = y[:, -1]

        if self.log:
            x[:, :, 1] = np.exp(x[:, :, 1]) - 1.e-4
        if self.erf_phi:
            x[:, :, 1] = scipy.special.erf(x[:, :, 1])*np.pi/0.999

        p_w1_eppp = jet2cartesian(x[:, 3]) + jet2cartesian(x[:, 4])
        p_w2_eppp = jet2cartesian(x[:, 8]) + jet2cartesian(x[:, 9])
        p_t1_eppp = p_w1_eppp + jet2cartesian(x[:, 1])
        p_t2_eppp = p_w2_eppp + jet2cartesian(x[:, 6])

        x[:, 0] = cartesian2jet(p_t1_eppp)
        x[:, 2] = cartesian2jet(p_w1_eppp)
        x[:, 5] = cartesian2jet(p_t2_eppp)
        x[:, 7] = cartesian2jet(p_w2_eppp)
        if self.hadronic_first:
            x = x[:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]
        return torch.tensor(x).to(p.dtype).to(p.device)


class cartesian_hard(PreprocTrafo):
    def __init__(
        self,
        mppp=True,
        drop_masses=False,
        hadronic_first=False,
        log=False
    ):
        self.mppp = mppp
        self.drop_masses = drop_masses
        self.hadronic_first = hadronic_first
        self.log = log

        n_particles = 10
        output_dim = 6*4-4 - int(drop_masses)

        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(output_dim,),
            invertible=True
        )

        self.m_b = 4.7


    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.naive_reverse(x)
        else:
            return self.naive_forward(x)

    def naive_forward(self, p):
        if self.hadronic_first:
            y = p.clone()[:, [6, 8, 9, 1, 3, 4]].detach().cpu().numpy()
        else:
            y = p.clone()[:, [1, 3, 4, 6, 8, 9]].detach().cpu().numpy()

        x = jet2cartesian(y)

        x[:, :, 0] = y[:, :, 0]

        #if self.log:
        #    x[:, :, 0] = np.log(x[:, :, 0]+1.e-4)
        x = x.reshape(x.shape[0], -1)
        # kick b mass, neutrino mass, bbar mass
        if not self.drop_masses:
            keep_channels = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23]
        else:
            keep_channels = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 21, 22, 23]
            keep_channels = keep_channels + [4] if self.hadronic_first else keep_channels + [16]
        x = x[:, keep_channels]
        return torch.tensor(x, dtype=p.dtype, device=p.device)

    def naive_reverse(self, p):
        y = p.clone().detach().cpu().numpy()
        x = np.zeros((p.shape[0], 10, 4))

        if not self.drop_masses:
            x[:, 1, 0] = self.m_b
            x[:, 1, 1:] = y[:, :3]
            x[:, 3] = y[:, 3:7]
            x[:, 4, 1:] = y[:, 7:10]
            x[:, 6, 0] = self.m_b
            x[:, 6, 1:] = y[:, 10:13]
            x[:, 8] = y[:, 13:17]
            x[:, 9, 1:] = y[:, 17:20]
        else:
            x[:, 1, 0] = self.m_b
            x[:, 1, 1:] = y[:, :3]
            x[:, 3, 1:] = y[:, 3:6]
            x[:, 4, 1:] = y[:, 6:9]
            x[:, 6, 0] = self.m_b
            x[:, 6, 1:] = y[:, 9:12]
            x[:, 8, 1:] = y[:, 12:15]
            x[:, 9, 1:] = y[:, 15:18]
            if self.hadronic_first:
                x[:, 3, 0] = y[:, -1]
            else:
                x[:, 8, 0] = y[:, -1]

        #if self.log:
        #    x[:, :, 0] = np.exp(x[:, :, 0]) - 1.e-4

        x[:, :, 0] = np.sqrt((x[:, :]**2).sum(-1))

        x[:, 2] = x[:, 3] + x[:, 4]
        x[:, 7] = x[:, 8] + x[:, 9]
        x[:, 0] = x[:, 2] + x[:, 1]
        x[:, 5] = x[:, 7] + x[:, 6]

        x = cartesian2jet(x)
        if self.hadronic_first:
            x = x[:, [5, 6, 7, 8, 9, 0, 1, 2, 3, 4]]
        return torch.tensor(x).to(p.dtype).to(p.device)


class ttbar_massparam(PreprocTrafo):
    def __init__(
        self,
        log=True,
        drop_masses=False,
        breit_wigner=False
    ):
        self.log = log
        self.drop_masses = drop_masses
        self.breit_wigner = breit_wigner

        n_particles = 10
        output_dim = 18

        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(output_dim,),
            invertible=True
        )

        self.m_b = 4.7


    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.pp_reverse(x)
        else:
            return self.pp_forward(x)

    def pp_forward(self, p):

        x = p.clone()
        dtype, device = p.dtype, p.device
        x = np.array(x[:, [0,2,3,5,7,8]].detach().cpu())
        x_eppp = jet2cartesian(x)
        t_decay = torch.tensor(apply_top_decay_mass_param_single(x_eppp[:, 0], x_eppp[:, 1], x_eppp[:, 2]), dtype=dtype)
        tbar_decay = torch.tensor(apply_top_decay_mass_param_single(x_eppp[:, 3], x_eppp[:, 4], x_eppp[:, 5]), dtype=dtype)
        if self.breit_wigner:
            t_decay[:, 0] = breit_wigner_forward(t_decay[:, 0], peak_position=172.7, width=1)
            t_decay[:, 4] = breit_wigner_forward(t_decay[:, 4], peak_position=80.4, width=1)

            tbar_decay[:, 0] = breit_wigner_forward(tbar_decay[:, 0], peak_position=172.7, width=1)
            tbar_decay[:, 4] = breit_wigner_forward(tbar_decay[:, 4], peak_position=80.4, width=1)

        return torch.cat([t_decay, tbar_decay], dim=1).to(dtype).to(device)

    def pp_reverse(self, p):
        y = np.array(p.clone().detach().cpu())
        if self.breit_wigner:
            y[:, 0] = breit_wigner_reverse(y[:, 0], peak_position=172.7, width=1)
            y[:, 4] = breit_wigner_reverse(y[:, 4], peak_position=80.4, width=1)

            y[:, 9] = breit_wigner_reverse(y[:, 9], peak_position=172.7, width=1)
            y[:, 13] = breit_wigner_reverse(y[:, 13], peak_position=80.4, width=1)
        dtype, device = p.dtype, p.device
        x = np.zeros((p.shape[0], 10, 4))

        x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4] = undo_top_decay_mass_param_single(y[:, :9])
        x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9] = undo_top_decay_mass_param_single(y[:, 9:])
        x = cartesian2jet(x)
        return torch.tensor(x, device=device, dtype=dtype)


class ttbar_massparam_v2(PreprocTrafo):
    def __init__(
        self,
        log=True,
        drop_masses=False,
        breit_wigner=False,
        mW_first=False
    ):
        self.log = log
        self.drop_masses = drop_masses
        self.breit_wigner = breit_wigner
        self.mW_first = mW_first

        n_particles = 10
        output_dim = 19

        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(output_dim,),
            invertible=True
        )

        self.m_b = 4.7

    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.pp_reverse(x)
        else:
            return self.pp_forward(x)

    def pp_forward(self, p):

        x = p.clone()
        dtype, device = p.dtype, p.device
        x = np.array(x[:, [0,2,3,5,7,8]].detach().cpu())
        x_eppp = jet2cartesian(x)
        t_decay = torch.tensor(apply_top_decay_mass_param_single(x_eppp[:, 0], x_eppp[:, 1], x_eppp[:, 2]), dtype=dtype)
        tbar_decay = torch.tensor(apply_top_decay_mass_param_single_hadronic(x_eppp[:, 3], x_eppp[:, 4], x_eppp[:, 5]), dtype=dtype)

        if self.breit_wigner:
            t_decay[:, 0] = breit_wigner_forward(t_decay[:, 0], peak_position=172.7, width=1)
            t_decay[:, 4] = breit_wigner_forward(t_decay[:, 4], peak_position=80.4, width=1)

            tbar_decay[:, 0] = breit_wigner_forward(tbar_decay[:, 0], peak_position=172.7, width=1)
            tbar_decay[:, 4] = breit_wigner_forward(tbar_decay[:, 4], peak_position=80.4, width=1)


        full_massparam = torch.cat([t_decay, tbar_decay], dim=1).to(dtype).to(device)

        if self.mW_first:
            print("Putting mW first")
            changed_order = [4, 13, 0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12]
            full_massparam[:, :14] = full_massparam[:, changed_order]
        return full_massparam

    def pp_reverse(self, p):
        y = np.array(p.clone().detach().cpu())
        if self.mW_first:
            changed_order = [2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 1]
            y[:, :14] = y[:, changed_order]
        if self.breit_wigner:
            y[:, 0] = breit_wigner_reverse(y[:, 0], peak_position=172.7, width=1)
            y[:, 4] = breit_wigner_reverse(y[:, 4], peak_position=80.4, width=1)

            y[:, 9] = breit_wigner_reverse(y[:, 9], peak_position=172.7, width=1)
            y[:, 13] = breit_wigner_reverse(y[:, 13], peak_position=80.4, width=1)
        dtype, device = p.dtype, p.device
        x = np.zeros((p.shape[0], 10, 4))

        x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4] = undo_top_decay_mass_param_single(y[:, :9])
        x[:, 5], x[:, 6], x[:, 7], x[:, 8], x[:, 9] = undo_top_decay_mass_param_single_hadronic(y[:, 9:])
        x = cartesian2jet(x)
        return torch.tensor(x, device=device, dtype=dtype)


class transfermer_reco(PreprocTrafo):
    def __init__(
        self,
        n_jets=4,
        log=True,
        eta_cut=False
    ):
        n_particles = 2+n_jets
        self.log = log
        self.eta_cut = eta_cut
        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(n_particles, 4),
            invertible=True
        )


    def transform(
        self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.transfermer_reverse(x)
        else:
            return self.transfermer_forward(x)

    def transfermer_forward(self, p):
        x = p.clone()
        if self.log:
            x[:, :1, 1] = (x[:, :1, 1]+1.e-4).log()
            x[:, 1:, :2] = (x[:, 1:, :2] + 1.e-4).log()
        return x

    def transfermer_reverse(self, p):
        x = p.clone()
        if self.log:
            x[:, :1, 1] = (x[:, :1, 1]).exp() - 1.e-4
            x[:, 1:, :2] = (x[:, 1:, :2]).exp() - 1.e-4
        return x


class transfermer_hard(PreprocTrafo):
    def __init__(
            self,
            log=True,
            drop_masses=False,
            eta_cut=False
    ):

        self.log = log
        self.eta_cut = eta_cut
        super().__init__(
            input_shape=(10, 4),
            output_shape=(6, 4),
            invertible=True
        )

        self.drop_masses = drop_masses
        if self.drop_masses:
            self.masses = [None, 4.7, None, 0, 0, None, 4.7, None, 0, 0]

    def transform(
            self, x: torch.Tensor, rev: bool) -> torch.Tensor:
        if rev:
            return self.transfermer_reverse(x)
        else:
            return self.transfermer_forward(x)

    def transfermer_forward(self, p):
        y = p.clone().detach().cpu().numpy()[:, [1, 3, 4, 6, 8, 9]]
        if self.log:
            y[:, :, 1] = (y[:, :, 1] + 1.e-4).log()
        x = np.zeros((p.shape[0], 6, 4))
        x[:, :, 0] = y[:, :, 1]
        x[:, :, 1] = y[:, :, 2]
        x[:, :, 2] = y[:, :, 3]/np.pi
        x[:, :, 3] = y[:, :, 0]
        return torch.tensor(x, device=p.device, dtype=p.dtype)

    def transfermer_reverse(self, p):

        y = p.clone().detach().cpu().numpy()
        x = np.zeros((p.shape[0], 10, 4))
        x[:, [1, 3, 4, 6, 8, 9], 0] = y[:, :, 3]
        x[:, [1, 3, 4, 6, 8, 9], 1] = y[:, :, 0]
        x[:, [1, 3, 4, 6, 8, 9], 2] = y[:, :, 1]
        x[:, [1, 3, 4, 6, 8, 9], 3] = y[:, :, 2]*np.pi

        x[:, [2, 7], 0] = 4.7
        if self.drop_masses:
            x[:, [3, 4, 8, 9], 0] = 0

        if self.log:
            x[:, [1, 3, 4, 6, 8, 9], 1] = (x[:, [1, 3, 4, 6, 8, 9], 1]).exp() - 1.e-4

        p_w1_eppp = jet2cartesian(x[:, 3]) + jet2cartesian(x[:, 4])
        p_w2_eppp = jet2cartesian(x[:, 8]) + jet2cartesian(x[:, 9])
        p_t1_eppp = p_w1_eppp + jet2cartesian(x[:, 1])
        p_t2_eppp = p_w2_eppp + jet2cartesian(x[:, 6])

        x[:, 0] = cartesian2jet(p_t1_eppp)
        x[:, 2] = cartesian2jet(p_w1_eppp)
        x[:, 5] = cartesian2jet(p_t2_eppp)
        x[:, 7] = cartesian2jet(p_w2_eppp)
        return torch.tensor(x, device=p.device, dtype=p.dtype)