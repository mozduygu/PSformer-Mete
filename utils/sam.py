"""
SAM (Sharpness-Aware Minimization) optimizer wrapper.

Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
Improving Generalization", ICLR 2021. Used by PSformer per Appendix A.7
with the same formulation as SAMformer (Ilbert et al., ICML 2024).

Two-step procedure per training iteration:
  1. Compute gradient at theta. Let g = grad(L(theta)).
  2. Compute perturbation eps_hat = rho * g / ||g||.
  3. Re-compute gradient at (theta + eps_hat). Let g_hat = grad(L(theta + eps_hat)).
  4. Apply g_hat as the descent direction at the original theta.

In practice we use SAM as a wrapper around a base optimizer. The user must
implement their training loop with two forward+backward passes, calling
optimizer.first_step() between the first and second backward, and
optimizer.second_step() at the very end.

Standard usage in a training loop:

    optimizer.zero_grad()
    loss1 = criterion(model(x), y)         # forward pass 1
    loss1.backward()                       # backward pass 1
    optimizer.first_step(zero_grad=True)   # ascend to theta + eps_hat

    loss2 = criterion(model(x), y)         # forward pass 2 at perturbed weights
    loss2.backward()                       # backward pass 2
    optimizer.second_step(zero_grad=True)  # restore theta and apply update with g_hat
"""

import torch


class SAM(torch.optim.Optimizer):
    """
    SAM optimizer wrapper.

    Args:
        params:       model parameters (or param groups).
        base_optimizer: an UN-INSTANTIATED optimizer class (e.g., torch.optim.Adam).
        rho:          neighborhood radius rho (paper denotes this as the
                      'sharpness-aware' radius). Per-dataset values are listed
                      in Table 11 of the paper.
        adaptive:     if True, use ASAM (Kwon et al. 2021) adaptive rescaling
                      of the perturbation. Foret-style SAM (used by PSformer)
                      uses adaptive=False.
        **kwargs:     forwarded to the base optimizer's __init__.

    Notes:
        - We use the L2 norm across all parameters to compute the perturbation.
          (Foret et al. 2021 actually uses a *dual* p-norm; for p=2 the dual
          is also p=2, which is what every public implementation uses.)
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05,
                 adaptive: bool = False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"rho must be non-negative, got {rho}")
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        # Base optimizer holds the actual update logic; we share param_groups.
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Compute perturbation eps_hat and ascend: theta <- theta + eps_hat."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Adaptive (ASAM) variant -- not used by PSformer
                if group["adaptive"]:
                    e_w = (torch.pow(p, 2) * p.grad) * scale.to(p)
                else:
                    e_w = p.grad * scale.to(p)
                # Save for later (we'll subtract these in second_step)
                self.state[p]["e_w"] = e_w
                p.add_(e_w)
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Descend: theta <- (theta + eps_hat) - eps_hat = theta, then apply update."""
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "e_w" not in self.state[p]:
                    continue
                # Restore original parameters
                p.sub_(self.state[p]["e_w"])
        # Now apply the actual update using the gradients computed at theta+eps_hat
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def step(self, closure=None):
        """Convenience step that accepts a closure performing the two passes."""
        assert closure is not None, "SAM requires a closure that performs two backwards"
        closure = torch.enable_grad()(closure)
        loss = closure()
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
        return loss

    def _grad_norm(self) -> torch.Tensor:
        """L2 norm of the gradients across all parameters, on the same device."""
        # Pick a device that has params for placing the norm
        shared_device = self.param_groups[0]["params"][0].device
        norms = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if group["adaptive"]:
                    norms.append((torch.abs(p) * p.grad).norm(p=2).to(shared_device))
                else:
                    norms.append(p.grad.norm(p=2).to(shared_device))
        if not norms:
            return torch.tensor(0.0, device=shared_device)
        return torch.norm(torch.stack(norms), p=2)

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
