#!/usr/bin/env python3
# encoding: utf-8
# @Time    : 2018/8/1 上午1:50
# @Author  : yuchangqian
# @Contact : changqian_yu@163.com
# @File    : lr_policy.py.py

from abc import ABCMeta, abstractmethod
import math


class BaseLR():
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_lr(self, cur_iter): pass


class PolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0

    def get_lr(self, cur_iter):
        return self.start_lr * (
                (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class WarmUpPolyLR(BaseLR):
    def __init__(self, start_lr, lr_power, total_iters, warmup_steps):
        self.start_lr = start_lr
        self.lr_power = lr_power
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        else:
            return self.start_lr * (
                    (1 - float(cur_iter) / self.total_iters) ** self.lr_power)


class MultiStageLR(BaseLR):
    def __init__(self, lr_stages):
        assert type(lr_stages) in [list, tuple] and len(lr_stages[0]) == 2, \
            'lr_stages must be list or tuple, with [iters, lr] format'
        self._lr_stagess = lr_stages

    def get_lr(self, epoch):
        for it_lr in self._lr_stagess:
            if epoch < it_lr[0]:
                return it_lr[1]


class LinearIncreaseLR(BaseLR):
    def __init__(self, start_lr, end_lr, warm_iters):
        self._start_lr = start_lr
        self._end_lr = end_lr
        self._warm_iters = warm_iters
        self._delta_lr = (end_lr - start_lr) / warm_iters

    def get_lr(self, cur_epoch):
        return self._start_lr + cur_epoch * self._delta_lr


class CyclicLR(BaseLR):
    def __init__(self, min_lr, max_lr, cycle_epochs, warmup_epochs, total_iters, iters_per_epoch):
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cycle_epochs = cycle_epochs
        self.warmup_epochs = warmup_epochs
        self.total_iters = total_iters
        self.iters_per_epoch = iters_per_epoch
        self.min_momentum = 0.85
        self.max_momentum = 0.95

    def get_lr(self, cur_iter):
        current_epoch = cur_iter // self.iters_per_epoch
        
        # Warmup phase
        if current_epoch < self.warmup_epochs:
            return self.min_lr + (self.max_lr - self.min_lr) * (cur_iter / (self.warmup_epochs * self.iters_per_epoch))
        
        # After warmup, use cosine annealing with warm restarts
        current_epoch = current_epoch - self.warmup_epochs
        cycle = current_epoch // self.cycle_epochs
        cycle_epoch = current_epoch % self.cycle_epochs
        
        # Cosine annealing formula
        cos_progress = math.cos(math.pi * cycle_epoch / self.cycle_epochs)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + cos_progress)
        
        # Momentum varies inversely with learning rate
        momentum = self.max_momentum - 0.5 * (self.max_momentum - self.min_momentum) * (1 + cos_progress)
        return lr, momentum
    

class StepLR(BaseLR):
    def __init__(self, start_lr, step_size, gamma):
        self.start_lr = start_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, cur_epoch):
        return self.start_lr * (self.gamma ** (cur_epoch // self.step_size))
    

class CosineAnnealingWarmupLR(BaseLR):
    def __init__(self, start_lr, total_iters, warmup_steps, min_lr=0):
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.total_iters = total_iters + 0.0
        self.warmup_steps = warmup_steps

    def get_lr(self, cur_iter):
        # Warmup phase
        if cur_iter < self.warmup_steps:
            return self.start_lr * (cur_iter / self.warmup_steps)
        
        # Cosine annealing phase
        progress = (cur_iter - self.warmup_steps) / (self.total_iters - self.warmup_steps)
        return self.min_lr + 0.5 * (self.start_lr - self.min_lr) * (1 + math.cos(math.pi * progress))


class OneCycleLR(BaseLR):
    def __init__(self, start_lr, max_lr, total_iters, pct_start=0.3, div_factor=25., final_div_factor=10000.):
        self.start_lr = start_lr
        self.max_lr = max_lr if max_lr is not None else start_lr * div_factor
        self.min_lr = start_lr / div_factor
        self.final_lr = start_lr / final_div_factor
        self.total_iters = total_iters + 0.0
        self.pct_start = pct_start
        
        # Calculate phases
        self.step_up_iters = int(self.total_iters * self.pct_start)
        self.step_down_iters = self.total_iters - self.step_up_iters

    def get_lr(self, cur_iter):
        if cur_iter < self.step_up_iters:
            # First phase: increasing learning rate
            return self.min_lr + (self.max_lr - self.min_lr) * (cur_iter / self.step_up_iters)
        else:
            # Second phase: decreasing learning rate
            remaining_iter = cur_iter - self.step_up_iters
            cosine_decay = 0.5 * (1 + math.cos(math.pi * remaining_iter / self.step_down_iters))
            return self.final_lr + (self.max_lr - self.final_lr) * cosine_decay


class ReduceLROnPlateauLR(BaseLR):
    def __init__(self, start_lr, factor=0.1, patience=10, min_lr=0, threshold=1e-4, cooldown=0):
        self.start_lr = start_lr
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.cooldown = cooldown
        
        # State variables
        self.current_lr = start_lr
        self.best_loss = float('inf')
        self.wait_count = 0
        self.cooldown_counter = 0
    
    def step(self, metrics):
        """Called after each validation round to update the learning rate"""
        
        # Check if in cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return self.current_lr
        
        # Check if improvement is significant
        if metrics < self.best_loss - self.threshold:
            # Better performance - reset wait counter
            self.best_loss = metrics
            self.wait_count = 0
        else:
            # No significant improvement
            self.wait_count += 1
            if self.wait_count >= self.patience:
                # Reduce learning rate
                old_lr = self.current_lr
                self.current_lr = max(self.current_lr * self.factor, self.min_lr)
                self.wait_count = 0
                self.cooldown_counter = self.cooldown
        
        return self.current_lr
    
    def get_lr(self, cur_iter):
        # Note: This method is required by BaseLR but should not be used directly
        # with ReduceLROnPlateau. Use step() method with validation metrics instead.
        return self.current_lr
    

