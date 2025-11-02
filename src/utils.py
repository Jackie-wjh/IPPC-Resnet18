import torch, random, numpy as np

def set_seed(seed: int = 312):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum, self.n = 0.0, 0
    def update(self, val, k=1): self.sum += float(val)*k; self.n += k
    @property
    def avg(self): return self.sum / max(1, self.n)