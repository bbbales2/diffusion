#cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

cimport numpy
import numpy

cpdef func(float t, numpy.ndarray[numpy.float_t, ndim = 1] y, int N, float D, float b, float dx):
    cdef numpy.ndarray[numpy.float_t, ndim = 1] u, dudD, dudb, \
                                        du, ddudD, ddudb

    u = numpy.zeros(N)
    dudD = numpy.zeros(N)
    dudb = numpy.zeros(N)
    du = numpy.zeros(N)
    ddudD = numpy.zeros(N)
    ddudb = numpy.zeros(N)

    cdef int i

    for i in range(0, N):
        u[i] = y[i]
        dudD[i] = y[i + N]
        dudb[i] = y[i + 2 * N]

        du[i] = 0
        ddudD[i] = 0
        ddudb[i] = 0

    du[0] = D * (b - 2 * u[0] + u[1]) / dx**2

    ddudD[0] = du[0] / D + D * (-2 * dudD[0] + dudD[1]) / dx**2
    ddudb[0] = D / dx**2 + D * (-2 * dudb[0] + dudb[1]) / dx**2

    for i in range(1, N - 1):
        du[i] = D * (u[i - 1] - 2 * u[i] + u[i + 1]) / dx**2

        ddudD[i] = du[i] / D + D * (dudD[i - 1] - 2 * dudD[i] + dudD[i + 1]) / dx**2
        ddudb[i] = D * (dudb[i - 1] - 2 * dudb[i] + dudb[i + 1]) / dx**2

    du[N - 1] = D * (u[N - 2] - 2 * u[N - 1]) / dx**2
    ddudD[N - 1] = du[N - 1] / D + D * (dudD[N - 2] - 2 * dudD[N - 1]) / dx**2
    ddudb[N - 1] = D * (dudb[N - 2] - 2 * dudb[N - 1]) / dx**2

    return du#numpy.concatenate([, ddudD, ddudb])

cpdef jac(float t, numpy.ndarray[numpy.float_t, ndim = 1] y, int N, float D, float b, float dx):
    cdef numpy.ndarray[numpy.float_t, ndim = 2] ddu

    cdef int i

    ddu = numpy.zeros((N, N))

    ddu[0, 0] = -2 * D / dx**2
    ddu[0, 1] = D / dx**2

    for i in range(1, N - 1):
        ddu[i, i - 1] = D / dx**2
        ddu[i, i] = D / dx**2
        ddu[i, i + 1] = D / dx**2

    ddu[N - 1, N - 1] = -2 * D / dx**2
    ddu[N - 1, N - 2] = D / dx**2

    return ddu

cpdef numpy.ndarray[numpy.float_t, ndim = 1] solve(numpy.ndarray[numpy.float_t, ndim = 1] d, float off, numpy.ndarray[numpy.float_t, ndim = 1] b):
    cdef numpy.ndarray[numpy.float_t, ndim = 1] dt, bt

    cdef int i, N
    cdef float f, g

    N = d.shape[0]

    dt = d.copy()
    bt = b.copy()

    for i in range(N - 1):
        f = -off / dt[i]
        dt[i + 1] += off * f
        bt[i + 1] += f * bt[i]

    for i in range(N - 2, -1, -1):
        g = -off / dt[i + 1]
        bt[i] += g * bt[i + 1]

    return bt / dt