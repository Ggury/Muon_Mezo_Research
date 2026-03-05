import torch


def newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.float()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


def muon_update(grad, momentum, beta = 0.95, ms_steps = 5):
    momentum.lerp_(grad, 1 - beta)
    update = momentum
    if update.ndim >2:
        update = update.view(update.size(0), -1)
    elif update.ndim < 2:
        update = update.view(1,-1)
    update = newtonschulz5(update, steps=ms_steps)
    update*=max(1,update.size(-2) / update.size(-1))**0.5
    return update



class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr = 0.02, weight_decay = 0, momentum = 0.95):
        defaults = dict(lr = lr, weight_decay = weight_decay, momentum = momentum)
        super().__init__(params, defaults)
    @torch.no_grad()
    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p)
                update = muon_update(p.grad, state["momentum_buffer"], beta = group["momentum"])
                p.mul_(1-group["lr"] * group["weight_decay"])
                p.add_(update.reshape(p.shape), alpha =-group["lr"])
        return loss