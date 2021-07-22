import operator
from itertools import accumulate

import numpy as np

from torch.optim.lr_scheduler import _LRScheduler

class BaseLRScheduler(_LRScheduler):
    def __init__(self, 
        optimizer, 
        num_iter_per_epoch,
        num_epoch,
        last_epoch=-1):

        self.num_epoch  = num_epoch
        self.num_iter_per_epoch = num_iter_per_epoch
        self.last_epoch = last_epoch
        self.last_it    = 0
        
        super(BaseLRScheduler, self).__init__(optimizer, last_epoch)

        self.set_epoch_iter(0, 0)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def _update_lr(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

    def set_epoch_iter(self, new_epoch, new_it):
        self.last_epoch = new_epoch
        self.last_it    = new_it
        self._update_lr()

    def step(self, epoch=None):
        if epoch is not None:
            self.last_epoch = epoch
            self.last_it    = 0
            self._update_lr()

        else:
            self.optimizer.step()

            self.last_it += 1
            if self.last_it % self.num_iter_per_epoch == 0:
                self.last_it    = 0
                self.last_epoch+= 1

            self._update_lr()

    def get_lr(self):
        raise NotImplementedError

    @property
    def total_num_iter(self):
        return self.num_iter_per_epoch * self.num_epoch

class OneCycleLR(BaseLRScheduler):
    
    #def __init__(self, optimizer, milestones, last_epoch=-1):
    def __init__(self, 
        optimizer,
        num_it_per_epoch,
        num_epoch, 
        phases_ratio    = (0.15, 0.35, 0.35, 0.15),
        min_lr_factor   = 0.05,
        anneal_lr_factor= 0.01,
        **kwargs,
    ):

        if sum(phases_ratio) != 1.0 and len(phases_ratio) != 4:
            raise ValueError("phases_ratio must have length of 4 and sum of 1.0")
        else:
            self.phases_ratio = phases_ratio

        if min_lr_factor > 1.0:
            ValueError("Minimum LR factor must be less than or equal 1.0")
        else:
            self.min_lr_factor = min_lr_factor

        if anneal_lr_factor > 1.0:
            ValueError("Annealing LR factor must be less than or equal 1.0")
        else:
            self.anneal_lr_factor = anneal_lr_factor
        
        self.phases_step = [0]
        for total_ratio in accumulate(phases_ratio[:-1], operator.add):
            self.phases_step.append(
                round(total_ratio * num_it_per_epoch * num_epoch)
            )
        self.phases_step.append(
            num_it_per_epoch * num_epoch
        )        

        self.lr_factors = [
            min_lr_factor, # min
            1.0,           # max
            1.0,           # max
            min_lr_factor, # min
            min_lr_factor * anneal_lr_factor # annealing
        ]

        super(OneCycleLR, self).__init__(optimizer, num_it_per_epoch, num_epoch)

    def get_lr(self):

        curr_step = self.last_it + self.last_epoch * self.num_iter_per_epoch
        curr_step = curr_step % (self.num_iter_per_epoch * self.num_epoch)

        ###
        # Linear equation y = mx + c
        # m = (y_end - y_start) / (x_end - x_start)
        # x is relative step
        # Gradient m and bias c depend on phase
        for phase_idx, transition_step in enumerate(self.phases_step[1:]):
            if curr_step < transition_step:
                m = (self.lr_factors[phase_idx+1]  - self.lr_factors[phase_idx]) / \
                    (self.phases_step[phase_idx+1] - self.phases_step[phase_idx])
                c = self.lr_factors[phase_idx]
                x = curr_step - self.phases_step[phase_idx]
                lr_factor = m*x + c
                break

        return [base_lr * lr_factor  for base_lr in self.base_lrs]

class LogLR(BaseLRScheduler):
    def __init__(
        self, 
        optimizer,
        num_iter_per_epoch,
        num_epoch, 
        start_lr    = 1e-7,
        end_lr      = 1e1,
        kwargs      = None
    ):
        if kwargs is not None:
            try:
                start_lr = kwargs["start_lr"]
            except:
                pass
            
            try:
                end_lr = kwargs["end_lr"]
            except:
                pass
                
        self.start_lr   = start_lr
        self.end_lr     = end_lr
        self.num_epoch  = num_epoch

        super(LogLR, self).__init__(optimizer, num_iter_per_epoch, num_epoch)

    def get_lr(self):

        bias                  = np.log10(self.start_lr)
        gradient_numerator    = np.log10(self.end_lr) - np.log10(self.start_lr)
        gradient_denominator  = self.num_iter_per_epoch * self.num_epoch - 1

        current_lr = 10 ** (
                        bias + \
                        (self.last_it + self.last_epoch * self.num_iter_per_epoch) * \
                        gradient_numerator / \
                        gradient_denominator
                    )

        return [current_lr  for base_lr in self.base_lrs]