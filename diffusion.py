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
T = 1.0

D = 1.0

def solve(D):
    u = numpy.zeros(N)
    g = 0
    dudD = numpy.zeros(N)

    t = 0.0

    while t < T:
        u = u + dt * D * (A.dot(u) + b) / dx**2

        dudD = dudD + dt * ((A.dot(u) + b) / dx**2 + D * (A.dot(dudD)) / dx**2)

        t += dt

    #dfu = numpy.zeros(N)
    #dfu[0] = 1.0

    #dgu = -D * A / dx**2

    #dgp = -(A.dot(u) + b) / dx**2

    #lba = numpy.linalg.solve(dgu, -dfu)

    #print -lba.dot(dgp)
    #print dgu
    #print dgp
    #print dfu
    #print '---'

    return u, dudD

u1, g = solve(1.0)
u2, g = solve(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g
print u1
#%%
dt = 0.001
def solve(D):
    u = numpy.zeros(N)
    ut = numpy.zeros(N)
    g = 0
    dudD = numpy.zeros(N)

    t = 0.0

    alpha = dt * D / dx**2

    while t < T:
        ut = numpy.linalg.solve(numpy.eye(N) - alpha * A, u + alpha * b)

        d = numpy.ones(N) * (1 + 2 * alpha)
        bt = u + alpha * b
        M = numpy.eye(N) - alpha * A

        for i in range(N - 1):
            f = alpha / d[i]
            d[i + 1] -= alpha * f
            bt[i + 1] += f * bt[i]
            M[i + 1, :] += f * M[i]

        #plt.imshow(M, interpolation = 'NONE')
        #plt.show()

        for i in range(N - 2, -1, -1):
            g = alpha / d[i + 1]
            bt[i] += g * bt[i + 1]
            M[i, :] += g * M[i + 1]

        #plt.imshow(M, interpolation = 'NONE')
        #plt.show()

        u = bt / d
        #1/0

        dudD = dudD + dt * ((A.dot(u) + b) / dx**2 + D * (A.dot(dudD)) / dx**2)

        t += dt

    #dfu = numpy.zeros(N)
    #dfu[0] = 1.0

    #dgu = -D * A / dx**2

    #dgp = -(A.dot(u) + b) / dx**2

    #lba = numpy.linalg.solve(dgu, -dfu)

    #print -lba.dot(dgp)
    #print dgu
    #print dgp
    #print dfu
    #print '---'

    return u, ut, dudD

u1, ut1, g = solve(1.0)
u2, ut2, g = solve(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g
#%%
def solve2(D):
    u = numpy.zeros(N)
    g = numpy.zeros((1, N))

    def func(y, t):
        u = y[:N]
        g = y[N:]

        dy = numpy.zeros(2 * N)

        dy[0] = D * (b[0] - 2 * u[0] + u[1]) / dx**2
        dy[N] = dy[0] / D + D * (-2 * g[0] + g[1]) / dx**2

        for i in range(1, N - 1):
            dy[i] = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2
            dy[N + i] = dy[i] / D + D * (g[i - 1] - 2 * g[i] + g[i + 1]) / dx**2

        dy[N - 1] = D * (u[N - 2] - 2 * u[N - 1]) / dx**2
        dy[2 * N - 1] = dy[N - 1] / D + D * (g[N - 2] - 2 * g[N - 1]) / dx**2

        return dy#numpy.concatenate([D * (A.dot(u) + b) / dx**2, ((A.dot(u) + b) / dx**2 + D * (A.dot(g)) / dx**2)])

    _, y = scipy.integrate.odeint(func, numpy.zeros(2 * N), [0.0, T])

    u = y[:N]
    g = y[N:]

    return u, g

u1, g = solve2(1.0)
u2, g = solve2(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g
print u1
#%%
# This comes from Direct and adjoint sensitivity analysis ofchemical kinetic systems with KPP: Part I—theory and software tools
# Adrian Sandua, Dacian N. Daescub, and Gregory R. Carmichaelc
def solve(D):
    M = int(T / dt)
    u = numpy.zeros((M, N))
    g = numpy.zeros(M)

    for i in range(1, M):
        dudt = D * (A.dot(u[i - 1]) + b) / dx**2

        u[i] = u[i - 1] + dt * dudt

        g[i] = g[i - 1] + dt * dudt[0]

    l = numpy.zeros((M, N))

    l[M - 1, 0] = 1.0

    for i in range(M - 2, -1, -1):
        l[i] = l[i + 1] + dt * D * A.dot(l[i + 1]) / dx**2

    total = 0.0
    for i in range(M):
        dfdp = (A.dot(u[i]) + b) / dx**2

        total += dt * dfdp.dot(l[i])

    print total

    return u, total

u1, g = solve(1.0)
u2, g = solve(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g
#%%

ut = numpy.zeros(N + 2)
ut[1:-1] = solve(1.0)
ut[0] = b[0]
ut[-1] = b[-1]
plt.plot(x, ut, '*')
plt.show()