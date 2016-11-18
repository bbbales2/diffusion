#%%

import numpy
import matplotlib.pyplot as plt

N = 20
xs = numpy.linspace(0, 1, N + 2)
dx = xs[1] - xs[0]

c = numpy.zeros(20)
D = 1.3
a = 0.001

dt = 0.001

b1 = 1.0
b2 = 0.0

legend = []
#for a in [-0.0001, 0.0, 0.0001]:
    for t in range(20):
        co = c.copy()

        c[0] = co[0] + dt * (D * (co[1] - 2 * c[0] + b1) / dx**2 + D * (a / 2.0) * (co[1]**2 - 2 * c[0]**2 + b1**2) / dx**2)

        for i in range(1, N - 1):
            c[i] = co[i] + dt * (D * (co[i + 1] - 2 * c[i] + c[i - 1]) / dx**2 + D * (a / 2.0) * (co[i + 1]**2 - 2 * c[i]**2 + c[i - 1]**2) / dx**2)

        c[N - 1] = co[N - 1] + dt * (D * (b2 - 2 * c[N - 1] + c[N - 2]) / dx**2 + D * (a / 2.0) * (b2**2 - 2 * c[N - 1]**2 + c[N - 2]**2) / dx**2)

    plt.plot(xs[1 : -1], c)
    legend.append("a = {0}".format(a))
plt.legend(legend)
plt.show()