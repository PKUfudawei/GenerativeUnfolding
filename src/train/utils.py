import torch
import numpy as np
from scipy.stats import wasserstein_distance
from scipy.special import erf, erfinv

# adapted from https://github.com/ViniciusMikuni/SBUnfold
def GetEMD(ref, array, weights_arr=None, nboot=100):
    if weights_arr is None:
        weights_arr = np.ones(len(array))
    ds = []

    if nboot > 0:
        for _ in range(nboot):
            arr_idx = np.random.choice(range(array.shape[0]), array.shape[0])
            array_boot = array[arr_idx]
            w_boot = weights_arr[arr_idx]
            ds.append(10*wasserstein_distance(ref, array_boot, v_weights=w_boot))

        return np.mean(ds), np.std(ds)
    else: # no bootstrapping
        EMD = 10*wasserstein_distance(ref, array, v_weights=weights_arr)
        return EMD


# adapted from https://github.com/ViniciusMikuni/SBUnfold
def get_triangle_distance(true, predicted, bins, weights=None, nboot=100):
    x, _ = np.histogram(true, bins=bins, density=True)
    ds = []

    if nboot > 0:
        for _ in range(nboot):
            arr_idx = np.random.choice(range(predicted.shape[0]), predicted.shape[0])
            predicted_boot = predicted[arr_idx]
            y, _ = np.histogram(predicted_boot, bins=bins, density=True, weights=weights)
            dist = 0
            w = bins[1:] - bins[:-1]
            for ib in range(len(x)):
                dist+=0.5*w[ib]*(x[ib] - y[ib])**2/(x[ib] + y[ib]) if x[ib] + y[ib] >0 else 0.0
            ds.append(dist*1e3)
        return np.mean(ds), np.std(ds)
    else:
        y, _ = np.histogram(predicted, bins=bins, density=True, weights=weights)
        dist = 0
        w = bins[1:] - bins[:-1]
        for ib in range(len(x)):
            dist+=0.5*w[ib]*(x[ib] - y[ib])**2/(x[ib] + y[ib]) if x[ib] + y[ib] >0 else 0.0
        return dist*1e3



# taken from https://github.com/ViniciusMikuni/SBUnfold
def get_normalisation_weight(len_current_samples, len_of_longest_samples):
    return np.ones(len_current_samples) * (len_of_longest_samples / len_current_samples)


# applies a lorentz boost to p_target, whose velocity is inferred
# from p_frame_particle, such that the new frame becomes the rest
# frame of the particle with momentum p_frame_particle;
# note that this algorithm is numerically unstable to a significant
# extent for highly relativistic events
def lorentz_boost(p_target, p_frame_particle):
    K = np.array([
        [[0, 1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 1, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 1],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [1, 0, 0, 0]],
    ])

    beta = p_frame_particle[...,1:] / p_frame_particle[...,0:1]

    # don't boost momenta when norm(beta) is zero
    boost_mask = np.linalg.norm(beta, axis=-1) != 0.0
    n = beta[boost_mask] / np.linalg.norm(beta[boost_mask], axis=-1, keepdims=True)
    rapidity = np.arctanh(np.sqrt(np.sum(beta[boost_mask]**2, axis=-1)))

    n = np.expand_dims(n, axis=(-1,-2))
    rapidity = np.expand_dims(rapidity, axis=(-1,-2))

    n_dot_K = 0
    for i in range(len(K)):
        n_dot_K += n[...,i,:,:] * K[i]

    B = np.eye(4, 4) - np.sinh(rapidity) * n_dot_K \
        + (np.cosh(rapidity) - 1) * (n_dot_K @ n_dot_K)

    p_boosted = np.full_like(p_target, np.nan)
    p_boosted[boost_mask] = \
        (B @ np.expand_dims(p_target[boost_mask], axis=-1)).squeeze(-1)
    p_boosted[~boost_mask] = p_target[~boost_mask]
    return p_boosted

def rotate(p, p_new_z_axis):
    theta = polar_angle(p_new_z_axis)[...,None,None]

    u = np.cross(p_new_z_axis[..., 1:], [0, 0, 1])
    u = u / norm(u, keepdims=True)
    u_cross = np.cross(u[...,None,:], -np.eye(3)) # compute cross product matrix
    u_outer_prod = np.einsum('...i,...j', u, u)

    R = np.cos(theta) * np.eye(3) + np.sin(theta) * u_cross \
        + (1 - np.cos(theta)) * u_outer_prod

    p_new = np.full_like(p, np.nan)
    p_new[..., 0] = p[..., 0]
    p_new[..., 1:] = (R @ np.expand_dims(p[..., 1:], axis=-1)).squeeze(-1)
    return p_new
def unrotate(p, p_z_axis_old_frame):
    p_new_z_axis = np.full_like(p_z_axis_old_frame, np.nan)
    p_new_z_axis[..., 0] = p_z_axis_old_frame[..., 0]
    p_new_z_axis[..., 1] = -p_z_axis_old_frame[..., 1]
    p_new_z_axis[..., 2] = -p_z_axis_old_frame[..., 2]
    p_new_z_axis[..., 3] = p_z_axis_old_frame[..., 3]
    return rotate(p, p_new_z_axis)

def dot(p1, p2):
    return p1[..., 0] * p2[..., 0] - np.sum(p1[..., 1:] * p2[..., 1:], axis=-1)

def energy(p):
    return p[...,0]

def momentum_x(p):
    return p[...,1]

def momentum_y(p):
    return p[...,2]

def momentum_z(p):
    return p[...,3]

def invariant_mass(p):
    # compute sqrt(p0**2 - p1**2 - p2**3 - p3**3)
    m2 = invariant_mass_squared(p)
    return np.sqrt(np.clip(m2, 0, None))

def invariant_mass_squared(p):
    # compute p0**2 - p1**2 - p2**3 - p3**3
    return p[...,0]**2 - np.sum(p[...,1:]**2, axis=-1)

def mass_squared(p):
    return p[...,0]**2 - np.sum(p[...,1:]**2, axis=-1)

def mass(p, clip=True):
    mass_2 = mass_squared(p)
    if len(mass_2[mass_2 < 0]) > 0:
        print("Mass array shape:", mass_2.shape)
        print("Mass array negative masses ratio:", (mass_2 < 0).mean())
        print("Mass array negative masses ratio <-0.1:", (mass_2 < -0.1).mean())
        print("Mass array negative masses ratio <-1:", (mass_2 < -1).mean())
        print("Mass array negative masses mean:", mass_2[mass_2 < 0].mean(0))
        print("Mass array negative masses min:", mass_2[mass_2 < 0].min(0))
        print("Mass array negative masses std:", mass_2[mass_2 < 0].std(0))

    if clip:
        return np.sqrt(np.clip(mass_2, a_min=0, a_max=None))
    else:
        return np.sqrt(mass_2)

def norm_squared(p, keepdims=False):
    return np.sum(p[..., 1:]**2, axis=-1, keepdims=keepdims)

def norm(p, keepdims=False):
    return np.sqrt(norm_squared(p, keepdims=keepdims))

def pt2(p):
    return p[...,1]**2 + p[...,2]**2
def pt(p):
    return np.sqrt(pt2(p))
def pt2_from_norm(p_norm_squared, eta):
    return p_norm_squared * (1.0 - np.tanh(eta)**2)
def pt_from_norm(p_norm, eta):
    return p_norm * np.sqrt(1.0 - np.tanh(eta)**2)

def eta(p):
    # p_norm = np.sqrt(np.sum(p[...,1:]**2, axis=-1))
    #return 0.5 * np.log((p[..., 0] + p[..., 3]) / (p[..., 0] - p[..., 3]))
    return np.arctanh(p[..., 3] / np.sqrt(p[..., 1] ** 2 + p[..., 2] ** 2 + p[..., 3] ** 2))
def polar_angle(p):
    return np.arccos(p[..., 3] / norm(p))
def azimuthal_angle(p):
    return np.arctan2(p[...,2], p[...,1])

def pseudorapidity_diff(p_t1, p_t2):
    return eta(p_t2) - eta(p_t1)
def azimuthal_angle_diff(p_t1, p_t2):
    # TODO: [-2 pi, 2 pi] -> [-pi, pi]
    delta_phi =  azimuthal_angle(p_t2) - azimuthal_angle(p_t1)
    delta_phi[delta_phi > np.pi] = delta_phi[delta_phi > np.pi] - 2 * np.pi
    delta_phi[delta_phi < -np.pi] = delta_phi[delta_phi < -np.pi] + 2 * np.pi
    return delta_phi
def b4(p_top, p_antitop):
    return p_top[..., 3] * p_antitop[..., 3] \
        / (norm(p_top) * norm(p_antitop))
# azimuthal angle difference between two decay products of a top quark in the top
# pair rest frame
def azimuthal_angle_diff_cm(p_t1, p_t2, p1, p2):
    p_tt = p_t1 + p_t2
    p_top_tt = lorentz_boost(p_t1, p_tt)
    p1_tt = lorentz_boost(p1, p_tt)
    p2_tt = lorentz_boost(p2, p_tt)

    cross_12 = np.cross(p1_tt[..., 1:], p2_tt[..., 1:], axis=-1)
    cross_t1 = np.cross(p_top_tt[..., 1:], p1_tt[..., 1:], axis=-1)
    cross_t2 = np.cross(p_top_tt[..., 1:], p2_tt[..., 1:], axis=-1)

    cross_t1_t2_norm_prod = norm(cross_t1) * norm(cross_t2)
    cross_t1_t2_norm_prod[cross_t1_t2_norm_prod == 0] = 1e-8

    sign = np.sign(np.sum(p_top_tt[..., 1:] * cross_12, axis=-1))

    return sign * np.arccos(np.sum(cross_t1 * cross_t2, axis=-1) \
            / (cross_t1_t2_norm_prod))

def azimuthal_angle_diff_pm(p_tl, p_th, p_A, p_B):
    p_tt = p_tl + p_th
    p_tl_tt_unrotated = lorentz_boost(p_tl, p_tt)
    p_th_tt_unrotated = lorentz_boost(p_th, p_tt)
    p_A_tt_unrotated = lorentz_boost(p_A, p_tt)
    p_B_tt_unrotated = lorentz_boost(p_B, p_tt)

    p_tl_tt = rotate(p_tl_tt_unrotated, p_tl_tt_unrotated)
    p_th_tt = rotate(p_th_tt_unrotated, p_tl_tt_unrotated)
    p_A_tt = rotate(p_A_tt_unrotated, p_tl_tt_unrotated)
    p_B_tt = rotate(p_B_tt_unrotated, p_tl_tt_unrotated)

    need_AB_swap = dot(p_tl_tt - p_th_tt, p_A_tt - p_B_tt) < 0
    p_plus_tt = np.full_like(p_A_tt, np.nan)
    p_plus_tt[~need_AB_swap] = p_A_tt[~need_AB_swap]
    p_plus_tt[need_AB_swap] = p_B_tt[need_AB_swap]
    p_minus_tt = np.full_like(p_B_tt, np.nan)
    p_minus_tt[~need_AB_swap] = p_B_tt[~need_AB_swap]
    p_minus_tt[need_AB_swap] = p_A_tt[need_AB_swap]

    phi_plus_tt = azimuthal_angle(p_plus_tt)
    phi_minus_tt = azimuthal_angle(p_minus_tt)
    delta_phi_pm = phi_plus_tt - phi_minus_tt
    delta_phi_pm[delta_phi_pm > np.pi] = \
        delta_phi_pm[delta_phi_pm > np.pi] - 2 * np.pi
    delta_phi_pm[delta_phi_pm < -np.pi] = \
        delta_phi_pm[delta_phi_pm < -np.pi] + 2 * np.pi

    return delta_phi_pm

def levi_civita_product(p_tl, p_th, p_A, p_B):
    p_tt = p_tl + p_th
    p_tl_tt_unrotated = lorentz_boost(p_tl, p_tt)
    p_th_tt_unrotated = lorentz_boost(p_th, p_tt)
    p_A_tt_unrotated = lorentz_boost(p_A, p_tt)
    p_B_tt_unrotated = lorentz_boost(p_B, p_tt)

    p_tl_tt = rotate(p_tl_tt_unrotated, p_tl_tt_unrotated)
    p_th_tt = rotate(p_th_tt_unrotated, p_tl_tt_unrotated)
    p_A_tt = rotate(p_A_tt_unrotated, p_tl_tt_unrotated)
    p_B_tt = rotate(p_B_tt_unrotated, p_tl_tt_unrotated)

    need_AB_swap = dot(p_tl_tt - p_th_tt, p_A_tt - p_B_tt) < 0
    p_plus_tt = np.full_like(p_A_tt, np.nan)
    p_plus_tt[~need_AB_swap] = p_A_tt[~need_AB_swap]
    p_plus_tt[need_AB_swap] = p_B_tt[need_AB_swap]
    p_minus_tt = np.full_like(p_B_tt, np.nan)
    p_minus_tt[~need_AB_swap] = p_B_tt[~need_AB_swap]
    p_minus_tt[need_AB_swap] = p_A_tt[need_AB_swap]

    pt_plus_tt = pt(p_plus_tt)
    pt_minus_tt = pt(p_minus_tt)

    phi_plus_tt = azimuthal_angle(p_plus_tt)
    phi_minus_tt = azimuthal_angle(p_minus_tt)
    delta_phi_pm = phi_plus_tt - phi_minus_tt
    delta_phi_pm[delta_phi_pm > np.pi] = \
        delta_phi_pm[delta_phi_pm > np.pi] - 2 * np.pi
    delta_phi_pm[delta_phi_pm < -np.pi] = \
        delta_phi_pm[delta_phi_pm < -np.pi] + 2 * np.pi
    return 4 * p_tl_tt[..., 3] * p_tl_tt[..., 0] \
        * pt_plus_tt * pt_minus_tt * np.sin(delta_phi_pm)


# angle between top quark momentum and z-axis in the Collin-Soper frame which is a
# rest frame of the top quark pair, where the z-axis has equal angles to the beam
# momenta
def collins_soper_angle(p_top, p_antitop):
    p_parallel_boost = p_top + p_antitop
    p_parallel_boost[..., 1] = 0
    p_parallel_boost[..., 2] = 0

    # boost into the CS frame by two consecutive boosts: first boost along the beam
    # axis, such that the z-momentum of the top antitop system vanishes, second boost
    # the top antitop system into its rest frame such that the transverse momentum
    # vanishes as well
    p_top_after_first = lorentz_boost(p_top, p_parallel_boost)
    p_antitop_after_first = lorentz_boost(p_antitop, p_parallel_boost)

    p_transverse_boost = p_top_after_first + p_antitop_after_first
    p_transverse_boost[..., 3] = 0

    p_top_cs = lorentz_boost(p_top_after_first, p_transverse_boost)

    return np.arccos(p_top_cs[..., 3] / norm(p_top_cs))


def mother_particle(*particles):

    px_sum = 0
    py_sum = 0
    pz_sum = 0
    e_sum = 0
    for particle in particles:
        print(particle.shape)
        m = particle[..., 0]
        pT = particle[..., 1]
        eta = particle[..., 2]
        phi = particle[..., 3]

        px = pT * np.cos(phi)
        py = pT * np.sin(phi)
        pz = pT * np.sinh(eta)
        e = np.sqrt(m ** 2 + px ** 2 + py ** 2 + pz ** 2)

        px_sum += px
        py_sum += py
        pz_sum += pz
        e_sum += e

    return np.sqrt(np.clip(
        (e_sum)**2 - (px_sum)**2 - (py_sum)**2 - (pz_sum)**2, 0, None
    ))


def jet2cartesian(p_m_pt_eta_phi):

    m ,pt, eta, phi = p_m_pt_eta_phi[..., 0], p_m_pt_eta_phi[..., 1], p_m_pt_eta_phi[..., 2], p_m_pt_eta_phi[..., 3]

    p = np.full((*m.shape, 4), np.nan)
    p[..., 1] = pt * np.cos(phi)
    p[..., 2] = pt * np.sin(phi)
    p[..., 3] = pt * np.sinh(eta)

    p[..., 0] = np.sqrt(norm_squared(p) + m**2)
    return p


def jet2cartesian_single(m ,pt, eta, phi):

    p = np.full((*m.shape, 4), np.nan)
    p[..., 1] = pt * np.cos(phi)
    p[..., 2] = pt * np.sin(phi)
    p[..., 3] = pt * np.sinh(eta)

    p[..., 0] = np.sqrt(norm_squared(p) + m**2)
    return p


def cartesian2jet(p_eppp):

    p = np.full(p_eppp.shape, np.nan)
    p[..., 0] = mass(p_eppp)
    p[..., 1] = pt(p_eppp)
    p[..., 2] = eta(p_eppp)
    p[..., 3] = azimuthal_angle(p_eppp)

    return p


def apply_top_decay_mass_param_single(p_top, p_w, p_x1):
    # computes top decay mass parameterization, with events
    # which have the following layout on the last axis:
    # (b, x1 = W decay product 1, x2 = W decay product 2)
    p_x2 = p_w - p_x1
    p_b = p_top - p_w

    p_w_frame_top = lorentz_boost(p_w, p_top)
    p_x1_frame_w = lorentz_boost(p_x1, p_w)

    params_out = {}
    m_t = mass(p_top)
    pt_t = pt(p_top)
    eta_t = eta(p_top)
    phi_t = azimuthal_angle(p_top)

    # store m2_w_scaled instead of m2_w, because even m2_w >= 0.0 can still lead to
    # invalid events when m_t < m_w + m_b; if instead 1 >= m2_w / (m2_t - m2_b) >= 0,
    # then m_t >= m_w + m_b and m_w >= 0 is always ensured.
    m_w = mass(p_w_frame_top)
    eta_w = eta(p_w_frame_top)
    phi_w = azimuthal_angle(p_w_frame_top)

    eta_x1 = eta(p_x1_frame_w)
    phi_x1 = azimuthal_angle(p_x1_frame_w)

    return np.stack([m_t,
                           pt_t,
                           eta_t,
                           phi_t,
                           m_w,
                           eta_w,
                           phi_w,
                           eta_x1,
                           phi_x1], axis=1)


def undo_top_decay_mass_param_single(mass_param, return_all = True):
    # undoes top decay mass parameterization, with events
    # which have the following layout (9 params) on the last axis:
    # (top (4 params), w (3 params), x1 (2 params))


    # compute top momentum
    p_top = jet2cartesian(mass_param[:, :4])

    # compute w momentum in top rest frame
    pnorm2_w = \
        (mass_param[:, 0]**2 + 4.7**2 - mass_param[:, 4]**2)**2 / (4 * mass_param[:, 0]**2) - 4.7**2
    pt_w = pt_from_norm(np.sqrt(pnorm2_w), mass_param[:, 5])
    p_w = jet2cartesian_single(mass_param[:, 4], pt_w, mass_param[:, 5], mass_param[:, 6])

    # compute up momentum in w rest frame
    E_x1 = mass_param[:, 4] / 2
    m_x1 = np.zeros_like(E_x1)
    pt_x1 = pt_from_norm(E_x1, mass_param[:, 7])
    p_x1 = jet2cartesian_single(m_x1, pt_x1, mass_param[:, 7], mass_param[:, 8])

    p_w = lorentz_boost(p_w, flip(p_top))
    p_x1 = lorentz_boost(p_x1, flip(p_w))

    p_b = p_top - p_w
    p_x2 = p_w - p_x1
    if return_all:
        return p_top, p_b, p_w, p_x1, p_x2
    else:
        return p_top, p_w, p_x1


def flip(p):
    p_flipped = np.full(p.shape, np.nan)
    p_flipped[...,0] = p[...,0]
    p_flipped[...,1:] = -p[...,1:]
    return p_flipped


def apply_top_decay_mass_param_single_hadronic(p_top, p_w, p_x1):
    # computes top decay mass parameterization, with events
    # which have the following layout on the last axis:
    # (b, x1 = W decay product 1, x2 = W decay product 2)
    p_x2 = p_w - p_x1
    p_b = p_top - p_w

    p_w_frame_top = lorentz_boost(p_w, p_top)
    p_x1_frame_w = lorentz_boost(p_x1, p_w)

    params_out = {}
    m_t = mass(p_top)
    pt_t = pt(p_top)
    eta_t = eta(p_top)
    phi_t = azimuthal_angle(p_top)

    # store m2_w_scaled instead of m2_w, because even m2_w >= 0.0 can still lead to
    # invalid events when m_t < m_w + m_b; if instead 1 >= m2_w / (m2_t - m2_b) >= 0,
    # then m_t >= m_w + m_b and m_w >= 0 is always ensured.
    m_w = mass(p_w_frame_top)
    eta_w = eta(p_w_frame_top)
    phi_w = azimuthal_angle(p_w_frame_top)

    m_x1 = mass(p_x1_frame_w, clip=True)
    eta_x1 = eta(p_x1_frame_w)
    phi_x1 = azimuthal_angle(p_x1_frame_w)

    return np.stack([m_t,
                           pt_t,
                           eta_t,
                           phi_t,
                           m_w,
                           eta_w,
                           phi_w,
                           m_x1,
                           eta_x1,
                           phi_x1], axis=1)


def undo_top_decay_mass_param_single_hadronic(mass_param, return_all = True):
    # undoes top decay mass parameterization, with events
    # which have the following layout (9 params) on the last axis:
    # (top (4 params), w (3 params), x1 (2 params))

    # m_t = mass_param[:, 0]
    # pt_t = mass_param[:, 1]
    # eta_t = mass_param[:, 2]
    # phi_t = mass_param[:, 3]
    # m_w = mass_param[:, 4]
    # eta_w = mass_param[:, 5]
    # phi_w = mass_param[:, 6]
    # m_x1 = mass_param[:, 7]
    # eta_x1 = mass_param[:, 8]
    # phi_x = mass_param[:, 9]

    # compute top momentum
    p_top = jet2cartesian(mass_param[:, :4])

    # compute w momentum in top rest frame
    pnorm2_w = \
        (mass_param[:, 0]**2 + 4.7**2 - mass_param[:, 4]**2)**2 / (4 * mass_param[:, 0]**2) - 4.7**2
    pt_w = pt_from_norm(np.sqrt(pnorm2_w), mass_param[:, 5])
    p_w = jet2cartesian_single(mass_param[:, 4], pt_w, mass_param[:, 5], mass_param[:, 6])

    # compute up momentum in w rest frame
    m_x1 = mass_param[:, 7]
    pnorm2_x1 = \
        (mass_param[:, 4] ** 2 - mass_param[:, 7] ** 2) ** 2 / (4 * mass_param[:, 4] ** 2)
    pt_x1 = pt_from_norm(np.sqrt(pnorm2_x1), mass_param[:, 8])
    p_x1 = jet2cartesian_single(m_x1, pt_x1, mass_param[:, 8], mass_param[:, 9])

    p_w = lorentz_boost(p_w, flip(p_top))
    p_x1 = lorentz_boost(p_x1, flip(p_w))

    p_b = p_top - p_w
    p_x2 = p_w - p_x1
    if return_all:
        return p_top, p_b, p_w, p_x1, p_x2
    else:
        return p_top, p_w, p_x1


def breit_wigner_forward(events, peak_position, width):
    z1 = 1 / np.pi * np.arctan((events - peak_position) / width) + 0.5
    return np.sqrt(2) * erfinv(2 * z1 - 1)


def breit_wigner_reverse(events, peak_position, width):
    a = events/np.sqrt(2)
    a = erf(a)
    a = 0.5*(a+1)
    a = a -0.5
    a = a*np.pi
    a = np.tan(a)
    a = a*width + peak_position
    return a


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