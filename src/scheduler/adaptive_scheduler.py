import numpy as np

def linear_anneal(start, end, progress):
    """
    Linearly interpolate between start and end given progress in [0,1].
    """
    return start + (end - start) * progress

def cyclical_anneal(start, end, progress, half_cycle=5e-2):
    """
    Perform cyclical annealing between start and end.
    Progress is interpreted periodically with period 'half_cycle'.
    """
    cycle_progress = (progress % half_cycle) / half_cycle
    return linear_anneal(start, end, cycle_progress)

class DStepScheduler:
    def __init__(self, d_steps_rate_init=1, grace=5, thresh=0.6, beta=0.99, max_d_steps=5):
        """
        Implements an adaptive schedule for the number of D-steps per G-step.

        - d_steps_rate_init: Initial number of D-steps per G-step.
        - grace: Number of epochs before adjusting.
        - thresh: Threshold for discriminator accuracy/loss.
        - beta: EMA smoothing factor.
        - max_d_steps: Upper bound for D-steps.
        """
        self.d_steps_rate = d_steps_rate_init
        self.grace = grace
        self.thresh = thresh
        self.beta = beta
        self.max_d_steps = max_d_steps
        self.running_metric = 0.0
        self.d_step_count = 0
        self.g_step_count = 0

    def get_d_steps_rate(self):
        return self.d_steps_rate

    def d_step(self):
        self.d_step_count += 1

    def is_g_step_time(self):
        return self.d_step_count >= self.d_steps_rate

    def g_step(self, metric_value):
        self.g_step_count += 1
        self.running_metric = self.beta * self.running_metric + (1 - self.beta) * metric_value

        if self.g_step_count > self.grace and self.running_metric < self.thresh:
            self.d_steps_rate = min(self.d_steps_rate + 1, self.max_d_steps)

        self.d_step_count = 0  # Reset D-step count

class ConstDStepScheduler:
    def __init__(self, d_steps_rate=1, beta=0.99):
        self.d_steps_rate = d_steps_rate

    def get_d_steps_rate(self):
        return self.d_steps_rate

    def d_step(self):
        pass

    def is_g_step_time(self):
        return True

    def g_step(self, metric_value):
        pass