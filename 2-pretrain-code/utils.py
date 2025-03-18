import math

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, init_lr, base_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.init_lr = init_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.step_num = 0

        # Initialize the LR to 0
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.init_lr

    def step(self):
        self.step_num += 1
        lr = self._get_lr(self.step_num)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self, step):
        # 1) Warmup phase
        if step < self.warmup_steps:
            # return self.base_lr * float(step) / float(self.warmup_steps)
            progress = float(step) / float(self.warmup_steps)
            return self.init_lr + (self.base_lr - self.init_lr) * progress
        # 2) Cosine decay phase
        else:
            progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.base_lr - self.min_lr) * cosine
        