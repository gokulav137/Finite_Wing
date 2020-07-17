import numpy as np


class SimpleElliptical:
    def __init__(self, a0_val=2 * np.pi, alpha0_val=0):
        self.a0 = None
        self.a0_val = a0_val
        self.c = None
        self.alpha0 = None
        self.alpha0_val = alpha0_val

    def update_a0(self, n=40):
        self.a0 = self.a0_val * np.ones((1, n))

    def update_c(self, pts, mid_chord, span):
        self.c = mid_chord * np.sqrt(1 - 4 * np.square(pts / span))

    def update_alpha0(self, n):
        self.alpha0 = self.alpha0_val * np.ones((1, n))

    def get_analytical(self, alpha, pts, mid_chord, span, v_inf=10, sharpness=100):
        self.update_a0(sharpness)
        self.update_c(pts, mid_chord, span)
        gamma0 = 0.5 * self.a0_val * v_inf * (alpha - self.alpha0_val) / (
                    1 + 0.25 * self.a0_val * mid_chord / span) * mid_chord
        gamma = gamma0 * np.sqrt(1 - np.square(2 * pts / span))
        return gamma

    def get_variables(self, n, pts, mid_chord, span):
        self.update_a0(n)
        self.update_c(pts, mid_chord, span)
        self.update_alpha0(n)
        return self.a0, self.alpha0, self.c


class SimpleTapered:
    def __init__(self, t_ratio=1, a0_val=2 * np.pi, alpha0_val=0):
        self.t_ratio = t_ratio
        self.a0 = None
        self.a0_val = a0_val
        self.c = None
        self.alpha0 = None
        self.alpha0_val = alpha0_val

    def update_a0(self, n):
        self.a0 = self.a0_val * np.ones((1, n))

    def update_c(self, pts, mid_chord, span):
        self.c = 2 * mid_chord * (0.5 - ((1 - 1 / self.t_ratio) / span) * np.abs(pts))

    def update_alpha0(self, n):
        self.alpha0 = self.alpha0_val * np.ones((1, n))

    def get_variables(self, n, pts, mid_chord, span):
        self.update_a0(n)
        self.update_c(pts, mid_chord, span)
        self.update_alpha0(n)
        return self.a0, self.alpha0, self.c
