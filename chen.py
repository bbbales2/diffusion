#%%

import numpy
from numpy import zeros, linspace, array
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import seaborn

N = 100

x = linspace(0.0, 1.0, N)
c = zeros(N)
c0 = c.copy()
D = 2.0e-5
u0 = 1e-3
cc = 2.0
cinf = 0.0
dt = 0.1
dx = x[1] - x[0]

plt.plot(x, c)
plt.show()

#%%
# c' = f(c, t)
#
# c' = D * d^2c/dx^2 - u * dc/dx
# u = u0(1 - c(0, t))
# D * dc/dx(0, t) = u(c0 - cc)
#
# Using centered diff for advection term from:
# https://en.wikipedia.org/wiki/Numerical_solution_of_the_convection%E2%80%93diffusion_equation#Explicit_scheme
#
def f(c, t, D, u0):
    u = u0 * (1 - c[0])
    dcdx0 = u * (c[0] - cc)

    f = zeros(c.shape)
    f[0] = D * ((c[1] - c[0]) / dx - dcdx0) / (2.0 * dx) - u * dcdx0 / D
    for i in range(1, N - 1):
        f[i] = D * (c[i + 1] - 2 * c[i] + c[i - 1]) / dx**2 - u * (c[i + 1] - c[i - 1]) / (2 * dx)
    i = N - 1
    f[i] = D * (cinf - 2 * c[i] + c[i - 1]) / dx**2 - u * (cinf - c[i - 1]) / (2 * dx)

    return f

plt.plot(x, scipy.integrate.odeint(f, zeros(N), [0.0, 1000.0], args = (D, u0))[1, :])
plt.show()
#%%
for i in range(10000):
    c = c + dt * f(c, i * dt, D, u0)
#%%
def solve(x, D, u0):
    return scipy.integrate.odeint(f, zeros(N), [0.0, 100.0], args = (D, u0))[1, :]

y = solve(None, 1e-5, 1e-3) + numpy.random.randn(N) * 0.05

popt, pcov = scipy.optimize.curve_fit(solve, None, y, p0 = (1e-4, 1e-2))

#%%
p = numpy.random.multivariate_normal(popt, pcov, 1000)
plt.plot(p[:, 0], p[:, 1], 'o')
plt.xlabel('D')
plt.ylabel('u0')
plt.show()
