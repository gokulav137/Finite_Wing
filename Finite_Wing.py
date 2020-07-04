import numpy as np
import matplotlib.pyplot as plt
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
        return np.array([(j+0.5)*self.span/n-self.span/2 for j in range(0, n)])

    def get_pts(self, n):
        self.cord_pts = self.calc_cord_pts(n)
        theta_pts = np.arccos(-2*self.cord_pts/self.span)
        return theta_pts

    def get_m(self, pts):
        sin_vals = np.sin(pts).reshape(-1, 1)
        sinn_vals = self.get_sinn(pts)
        a0, _, c = self.w_type.get_variables(self.no_seg, self.cord_pts, self.mid_chord, self.span)
        m = sinn_vals*2*self.span+(np.divide(sinn_vals, sin_vals)*(0.5*a0*c).reshape(-1, 1))*np.arange(1,
                                                                                                       self.no_freq+1)
        return m

    def get_sinn(self, pts):
        thetan_vals = np.outer(pts, np.arange(1, self.no_freq + 1))
        return np.sin(thetan_vals)

    def inv_m(self):
        self.M_inv = scipy.linalg.pinv(self.M)

    def update_rhs(self, alpha):
        a0, alpha0, c = self.w_type.get_variables(self.no_seg, self.cord_pts, self.mid_chord, self.span)
        self.rhs = (0.5*a0*c*[alpha-alpha0]).reshape(-1, 1)

    def update_a(self):
        self.A = np.dot(self.M_inv, self.rhs.reshape(-1))

    def simulate(self, v_inf=10, sharpness=100):
        assert sharpness > 0, "sharpness value should be greater than 0"
        pts = self.get_pts(sharpness)
        theta_mat = self.get_sinn(pts)
        gamma = np.dot(theta_mat, self.A)*2*v_inf*self.span
        return gamma

    def simulate_analytical(self, v_inf=10, sharpness=100, alpha=0):
        assert sharpness > 0, "sharpness value should be greater than 0"
        assert hasattr(self.w_type, 'get_analytical'), "w_type has no analytical solution provide (get_analytical)"
        return self.w_type.get_analytical(alpha, self.calc_cord_pts(sharpness),
                                          self.mid_chord, self.span, v_inf, sharpness)

    def get_a(self):
        return self.A


class SimpleElliptical:
    def __init__(self, a0_val=2*np.pi, alpha0_val=0):
        self.a0 = None
        self.a0_val = a0_val
        self.c = None
        self.alpha0 = None
        self.alpha0_val = alpha0_val

    def update_a0(self, n=40):
        self.a0 = self.a0_val*np.ones((1, n))

    def update_c(self, pts, mid_chord, span):
        self.c = mid_chord*np.sqrt(1-4*np.square(pts/span))

    def update_alpha0(self, n):
        self.alpha0 = self.alpha0_val*np.ones((1, n))

    def get_analytical(self, alpha, pts, mid_chord, span, v_inf=10, sharpness=100):
        self.update_a0(sharpness)
        self.update_c(pts, mid_chord, span)
        gamma0 = 0.5*self.a0_val*v_inf*(alpha-self.alpha0_val)/(1+0.25*self.a0_val*mid_chord/span)*mid_chord
        gamma = gamma0*np.sqrt(1-np.square(2*pts/span))
        return gamma

    def get_variables(self, n, pts, mid_chord, span):
        self.update_a0(n)
        self.update_c(pts, mid_chord, span)
        self.update_alpha0(n)
        return self.a0, self.alpha0, self.c


class SimpleTapered:
    def __init__(self, t_ratio=1, a0_val=2*np.pi, alpha0_val=0):
        self.t_ratio = t_ratio
        self.a0 = None
        self.a0_val = a0_val
        self.c = None
        self.alpha0 = None
        self.alpha0_val = alpha0_val

    def update_a0(self, n):
        self.a0 = self.a0_val * np.ones((1, n))

    def update_c(self, pts, mid_chord, span):
        self.c = 2*mid_chord*(0.5-((1-1/self.t_ratio)/span)*np.abs(pts))

    def update_alpha0(self, n):
        self.alpha0 = self.alpha0_val * np.ones((1, n))

    def get_variables(self, n, pts, mid_chord, span):
        self.update_a0(n)
        self.update_c(pts, mid_chord, span)
        self.update_alpha0(n)
        return self.a0, self.alpha0, self.c


def plot_gamma(y_data, x_data, wing_type, legends):
    plt.ylabel("Bound circulation ()")
    plt.xlabel("Span wise axis (m)")
    for data in y_data:
        plt.plot(x_data, data)
    plt.legend(legends)
    plt.title("Bound Circulation for "+wing_type+" Chord distribution")
    plt.show()


def plot_a(dataset, wing_type, legends=None):
    plt.ylabel("Value of Fourier Series Coefficients")
    plt.xlabel("Fourier Series Coefficients")
    length = len(dataset)
    for k, data in enumerate(dataset):
        plt.bar(x=np.arange(1, dataset[0].size+1)+(k-int(length/2))/length, height=data, width=1/length)
    if legends is not None:
        plt.legend(legends)
    plt.xticks(ticks=range(1, dataset[0].size+1), labels=['A'+str(j) for j in range(1, dataset[0].size+1)])
    plt.axhline(0, color='black')
    plt.title("Fourier Series coefficients for " + wing_type + " Chord distribution")
    plt.show()


W_elliptic = FiniteWing(SimpleElliptical(a0_val=2*np.pi, alpha0_val=0))
W_elliptic.solve(alpha=5*np.pi/180, no_seg=1000, no_freq=40)

num_gamma = W_elliptic.simulate(v_inf=10, sharpness=40)
ana_gamma = W_elliptic.simulate_analytical(v_inf=10, sharpness=40, alpha=5*np.pi/180)
elliptical_a = W_elliptic.get_a()[:10]
cord_pts = W_elliptic.calc_cord_pts(n=40)

plot_gamma([num_gamma, ana_gamma], cord_pts, "Elliptical", ["Numerical Value", "Analytical Value"])
plot_a([elliptical_a], "Elliptical")

W_tapered = []
data_gamma = []
data_a = []

for i, tr in enumerate([1, 2, 3]):
    W_tapered.append(FiniteWing(SimpleTapered(t_ratio=tr, a0_val=2 * np.pi, alpha0_val=0)))
    W_tapered[i].solve(alpha=5*np.pi/180, no_seg=100, no_freq=40)
    data_gamma.append(W_tapered[i].simulate(v_inf=10, sharpness=40))
    data_a.append(W_tapered[i].get_a()[:10])
cord_pts = W_tapered[0].calc_cord_pts(n=40)

plot_gamma(data_gamma, cord_pts, "Tapered", ["Taper ratio: 1", "Taper ratio: 2", "Taper ratio: 3"])
plot_a(data_a, "Tapered", ["Taper ratio: 1", "Taper ratio: 2", "Taper ratio: 3"])
