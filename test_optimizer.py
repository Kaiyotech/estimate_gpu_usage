from torch.optim import Optimizer
import torch


class Adam(Optimizer):
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super().__init__(params, {"lr": lr})

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.steps = 0

        self.m = []
        self.v = []

    @torch.no_grad()
    def step(self, closure=None):
        self.steps += 1
        n_bytes = 1024 * 1024 * 302

        if self.steps == 1:
            print("BEFORE ALLOCATION", torch.cuda.memory_allocated("cuda:0"), "|", torch.cuda.memory_reserved("cuda:0"))

            m = torch.zeros((n_bytes // 4, 1), dtype=torch.float32, device="cuda:0")
            allocated = m.nelement() * m.element_size()
            self.m.append(m)
            print("ALLOCATED", allocated)
            print("AFTER ALLOCATION", torch.cuda.memory_allocated("cuda:0"), "|", torch.cuda.memory_reserved("cuda:0"))
            print()

