import numpy as np
from finite_wing import FiniteWing
import wings
from plot_functions import plot_gamma, plot_a

W_elliptic = FiniteWing(wings.SimpleElliptical(a0_val=2 * np.pi, alpha0_val=0))
W_elliptic.solve(alpha=5 * np.pi / 180, no_seg=40, no_freq=40)

num_gamma = W_elliptic.simulate(v_inf=10, sharpness=40)
ana_gamma = W_elliptic.simulate_analytical(v_inf=10, sharpness=40, alpha=5 * np.pi / 180)
elliptical_a = W_elliptic.get_a()[:10]
cord_pts = W_elliptic.calc_cord_pts(n=40)

plot_gamma([num_gamma, ana_gamma], cord_pts, "Elliptical", ["Numerical Value", "Analytical Value"], line_type=True)
plot_a([elliptical_a], "Elliptical")

W_tapered = []
data_gamma = []
data_a = []

for i, tr in enumerate([1, 2, 3]):
    W_tapered.append(FiniteWing(wings.SimpleTapered(t_ratio=tr, a0_val=2 * np.pi, alpha0_val=0)))
    W_tapered[i].solve(alpha=5 * np.pi / 180, no_seg=100, no_freq=40)
    data_gamma.append(W_tapered[i].simulate(v_inf=10, sharpness=40))
    data_a.append(W_tapered[i].get_a()[:10])
cord_pts = W_tapered[0].calc_cord_pts(n=40)

plot_gamma(data_gamma, cord_pts, "Tapered", ["Taper ratio: 1", "Taper ratio: 2", "Taper ratio: 3"])
plot_a(data_a, "Tapered", ["Taper ratio: 1", "Taper ratio: 2", "Taper ratio: 3"])
