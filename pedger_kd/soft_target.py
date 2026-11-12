"""Soft training targets.

PEdger (main.py, official):
    M  = sigmoid(z_rec * w + z_ns * (1 - w)),  w = u_rec / (u_rec + u_ns),  u = |sigmoid(z) - 0.5|
    Y~ = Y * (1 - eta) + M * eta,              eta = 0.8 * epoch / n_epochs   (Y~ = Y in epoch 0)

Variant A (ours, plan section 8.3):
    Y~ = alpha * Y + (1 - alpha) * ramp * [(1 - gamma) * M + gamma * T]   (the (1-alpha)*ramp part
         is "trust_models"; the rest of the weight stays on Y)
"""
import torch

from .agreement import human_trust, teacher_conf_trust


def pedger_eta(epoch, n_epochs):
    return 0.8 * float(epoch) / float(n_epochs)


def fuse_momentum(z_rec, z_ns):
    """PEdger's confidence-weighted fusion of the two momentum networks' fused logits."""
    u_rec = torch.abs(torch.sigmoid(z_rec) - 0.5)
    u_ns = torch.abs(torch.sigmoid(z_ns) - 0.5)
    w = u_rec / (u_rec + u_ns).clamp_min(1e-12)
    return torch.sigmoid(z_rec * w + z_ns * (1 - w))


def pedger_target(Y, M, eta):
    return Y * (1 - eta) + M * eta


def gamma_confidence(T, M, eps=1e-6):
    """gamma(p) proportional to the teacher's confidence relative to the self-ensemble's."""
    cT = torch.abs(2 * T - 1)
    cM = torch.abs(2 * M - 1)
    return cT / (cT + cM + eps)


def agreement_target(Y, M, T, alpha, gamma=0.5, ramp=1.0):
    if M is None and T is None:
        return Y
    if T is None:
        model_view = M
    elif M is None:
        model_view = T
    else:
        model_view = (1 - gamma) * M + gamma * T
    trust_models = (1 - alpha) * ramp
    return (1 - trust_models) * Y + trust_models * model_view


def compute_alpha(method, a_alpha, T, eta):
    mode = method.get("alpha_mode", "agreement")
    if mode == "agreement":
        return human_trust(a_alpha, method.get("alpha_min", 0.1))
    if mode == "eta":  # uniform over pixels, PEdger's schedule
        return torch.full_like(a_alpha, 1.0 - eta)
    if mode == "teacher_conf":
        if T is None:
            raise ValueError("alpha_mode=teacher_conf needs a teacher map")
        return teacher_conf_trust(T, method.get("teacher_conf_c", 0.8))
    if mode == "const":
        return torch.full_like(a_alpha, float(method["alpha_const"]))
    raise ValueError(f"unknown alpha_mode {mode!r}")


def build_target(method, Y, M, T, a_alpha, epoch, n_epochs):
    """Return the soft target for one micro-batch.

    method: the `method:` block of the config.
    M is None in epoch 0 (PEdger has no momentum networks yet).
    """
    eta = pedger_eta(epoch, n_epochs)
    kind = method.get("target", "pedger")
    if kind == "pedger":
        if epoch == 0 or M is None:
            return Y
        return pedger_target(Y, M, eta)
    if kind == "agreement":
        alpha = compute_alpha(method, a_alpha, T, eta)
        ramp = float(epoch) / float(n_epochs) if method.get("ramp", True) else 1.0
        g = method.get("gamma", 0.5)
        if g == "confidence":
            g = gamma_confidence(T, M) if (T is not None and M is not None) else 0.5
        return agreement_target(Y, M, T, alpha, gamma=g, ramp=ramp)
    raise ValueError(f"unknown target {kind!r}")
