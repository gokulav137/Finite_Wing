import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np


def plot_gamma(y_data, x_data, wing_type, legends, line_type=False):
    if line_type:
        lines = ["-", "--"]
    else:
        lines = [None]
    line_cycle = cycle(lines)
    plt.ylabel("Bound circulation (m\u00b2/s)")
    plt.xlabel("Span wise axis (m)")
    for data in y_data:
        plt.plot(x_data, data, linestyle=next(line_cycle))
    plt.legend(legends)
    plt.title("Bound Circulation for " + wing_type + " Chord distribution")
    plt.show()


def plot_a(dataset, wing_type, legends=None):
    plt.ylabel("Value of Fourier Series Coefficients")
    plt.xlabel("Fourier Series Coefficients")
    length = len(dataset)
    for k, data in enumerate(dataset):
        plt.bar(x=np.arange(1, dataset[0].size + 1) + (k - int(length / 2)) / length, height=data, width=1 / length)
    if legends is not None:
        plt.legend(legends)
    plt.xticks(ticks=range(1, dataset[0].size + 1), labels=['A' + str(j) for j in range(1, dataset[0].size + 1)])
    plt.axhline(0, color='black')
    plt.title("Fourier Series coefficients for " + wing_type + " Chord distribution")
    plt.show()
