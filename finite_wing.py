import numpy as np
import scipy.linalg


class FiniteWing:
    def __init__(self, w_type, mid_chord=0.1, span=1):
        self.w_type = w_type
        assert mid_chord > 0, "mid_chord value should be greater than 0"
        self.mid_chord = mid_chord
        assert span > 0, "span value should be greater than 0"
        self.span = span
        self.no_seg = None
        self.no_freq = None
        self.cord_pts = None
        self.seg_pts = None
        self.M = None
        self.rhs = None
        self.M_inv = None
        self.A = None

    def solve(self, alpha=0, no_seg=40, no_freq=40):
        assert no_seg > 0, "no_seg value should be greater than 0"
        assert no_seg >= no_freq, "no_seg value should be greater or equal to no_freq"
        self.no_seg = no_seg
        self.no_freq = no_freq
        self.seg_pts = self.get_pts(self.no_seg)
        self.M = self.get_m(self.seg_pts)
        self.inv_m()
        self.update_rhs(alpha)
        self.update_a()

    def calc_cord_pts(self, n):
        return np.array([(j + 0.5) * self.span / n - self.span / 2 for j in range(0, n)])

    def get_pts(self, n):
        self.cord_pts = self.calc_cord_pts(n)
        theta_pts = np.arccos(-2 * self.cord_pts / self.span)
        return theta_pts

    def get_m(self, pts):
        sin_values = np.sin(pts).reshape(-1, 1)
        sinn_values = self.get_sinn(pts)
        a0, _, c = self.w_type.get_variables(self.no_seg, self.cord_pts, self.mid_chord, self.span)
        m = sinn_values * 2 * self.span + (np.divide(sinn_values, sin_values) * (0.5 * a0 * c).reshape(-1, 1)) \
            * np.arange(1, self.no_freq + 1)
        return m

    def get_sinn(self, pts):
        thetan_vals = np.outer(pts, np.arange(1, self.no_freq + 1))
        return np.sin(thetan_vals)

    def inv_m(self):
        self.M_inv = scipy.linalg.pinv(self.M)

    def update_rhs(self, alpha):
        a0, alpha0, c = self.w_type.get_variables(self.no_seg, self.cord_pts, self.mid_chord, self.span)
        self.rhs = (0.5 * a0 * c * [alpha - alpha0]).reshape(-1, 1)

    def update_a(self):
        self.A = np.dot(self.M_inv, self.rhs.reshape(-1))

    def simulate(self, v_inf=10, sharpness=100):
        assert sharpness > 0, "sharpness value should be greater than 0"
        pts = self.get_pts(sharpness)
        theta_mat = self.get_sinn(pts)
        gamma = np.dot(theta_mat, self.A) * 2 * v_inf * self.span
        return gamma

    def simulate_analytical(self, v_inf=10, sharpness=100, alpha=0):
        assert sharpness > 0, "sharpness value should be greater than 0"
        assert hasattr(self.w_type, 'get_analytical'), "w_type has no analytical solution provide (get_analytical)"
        return self.w_type.get_analytical(alpha, self.calc_cord_pts(sharpness),
                                          self.mid_chord, self.span, v_inf, sharpness)

    def get_a(self):
        return self.A
