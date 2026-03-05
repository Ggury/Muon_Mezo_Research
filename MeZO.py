import torch

class MeZO(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-5, eps=1e-3):
        defaults = dict(lr=lr, eps=eps)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, model, batch):

        group = self.param_groups[0]
        lr = group['lr']
        eps = group['eps']

        r_seed = torch.randint(0, 2**32, (1,)).item()
        def  add_noise(scale):
            torch.manual_seed(r_seed)
            for group in self.param_groups:
                for p in group['params']:
                    z = torch.randn_like(p)
                    p.add_(z, alpha= scale)
        add_noise(eps)
        L1 = model(**batch).loss
        add_noise(eps * -2)
        L2 = model(**batch).loss
        add_noise(eps)
        grad = (L1-L2)/(2*eps)
        torch.manual_seed(r_seed)
        for group in self.param_groups:
            for p in group['params']:
                z = torch.randn_like(p)
                p.add_(z, alpha=-lr * grad)
        return (L2 + L1) / 2