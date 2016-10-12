#%%

import numpy
import matplotlib.pyplot as plt
import scipy

N = 10

X = 1.0
dx = X / N
x = numpy.linspace(0.0, X, N + 2)

A = numpy.eye(N, N, k = -1) - 2 * numpy.eye(N, N) + numpy.eye(N, N, k = 1)

b = numpy.zeros(10)

b[0] = 1.0

dt = 0.0001
T = 0.025

D = 1.0

def solve(D):
    u = numpy.zeros(N)
    g = numpy.zeros((1, N))

    t = 0.0

    while t < T:
        u = u + dt * D * (A.dot(u) + b) / dx**2
        g[0] = g[0] + dt * ((A.dot(u) + b) / dx**2 + D * (A.dot(g[0])) / dx**2)

        t += dt

    return u, g

u1, g = solve(1.0)
u2, g = solve(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g[0]
#%%
def solve2(D):
    u = numpy.zeros(N)
    g = numpy.zeros((1, N))

    t = 0.0

    def func(y, t):
        u = y[:N]
        g = y[N:]

        dy = numpy.zeros(2 * N)

        for i in range(1, N - 1):
            dy[i] = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2
            dy[N + i] = dy[i] / D

        return numpy.concatenate([D * (A.dot(u) + b) / dx**2, ((A.dot(u) + b) / dx**2 + D * (A.dot(g)) / dx**2)])

    _, y = scipy.integrate.odeint(func, numpy.zeros(2 * N), [0.0, T])

    u = y[:N]
    g = y[N:]

    return u, g

u1, g = solve2(1.0)
u2, g = solve2(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g
#%%

ut = numpy.zeros(N + 2)
ut[1:-1] = solve(1.0)
ut[0] = b[0]
ut[-1] = b[-1]
plt.plot(x, ut, '*')
plt.show()