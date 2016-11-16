#%%
import numpy
import scipy
import os
os.chdir('/home/bbales2/diffusion')
import pyximport
import time
pyximport.install(reload_support = True)

import matplotlib.pyplot as plt

data = """0	2.496643737
5.6568	2.760147955
12.0599	2.731482238
18.4631	2.59114926
24.1199	2.48375488
29.7768	2.432533019
36.1799	2.240920246
41.8368	2.232590513
48.2399	2.161828643
53.8967	2.059430461
60.2999	1.996965007
65.9567	1.887728518
72.3598	1.907844208
78.017	1.808829964
84.42	1.57142775
90.077	1.701358373
95.734	1.605451805
102.137	1.456515395
108.54	1.461104922
114.197	1.352972172
119.854	1.302182066
126.257	1.243359915
132.66	1.066671415
138.317	1.159614932
143.973	1.126754173
151.045	1.070932217
156.701	1.032455722
162.358	1.045294806
168.015	0.96000336
174.418	0.893994653
180.821	0.903897486
186.478	0.830675659
192.135	0.774701278
198.538	0.718758126
204.941	0.716032941
210.598	0.713599868
216.255	0.630989886
222.658	0.648870892
228.315	0.567879499
234.718	0.541446439
240.375	0.548667775
246.778	0.468353856
252.435	0.495793694
258.838	0.450233883
264.495	0.435563393
270.898	0.403205311
276.555	0.427295594
282.212	0.359591388
289.283	-0.582969908
294.94	0.335402222
300.597	0.309952966
306.254	0.300129552
313.325	0.293129426
318.981	0.293583798
324.638	0.239513564
330.295	0.255303141
336.698	0.220204387
343.101	0.225505408
348.758	0.227246057
354.415	0.154325388
360.818	0.199752451
367.221	0.169280343
372.878	0.176321289
378.535	0.167477769
384.938	0.145899825
390.595	0.125536605
396.998	0.12135882
402.655	0.114396878
409.058	0.11778856
414.715	0.118246595
421.118	0.102681041
426.775	0.085488536
433.178	0.070239495
438.835	0.093348714
445.238	0.088575438
451.641	0.077767637
457.298	0.078043344
462.955	0.071393113
468.612	0.041080937
475.683	0.058835859
481.34	0.048843231
486.997	0.051745413
492.654	0.06697217"""

dataList = []
for line in data.split('\n')[0::2]:
    d, at = line.split()

    d = float(d)
    at = float(at)

    if at > 0.0:
        dataList.append((d, at))

data = numpy.array(dataList)

xs = numpy.array(data[:, 0])
at = numpy.array(data[:, 1])

nxs = numpy.linspace(0.0, max(xs), len(xs))
at = numpy.interp(nxs, xs, at)
xs = nxs

maxxs = max(xs)
maxat = max(at)

xs /= max(xs)
at /= max(at)
#%%
import diffusion_f
#%%

    #(_, y), info = scipy.integrate.odeint(func, numpy.zeros(N), [0.0, T], full_output = True)
    #ode = scipy.integrate.ode(diffusion_f.func, diffusion_f.jac)
    #ode.set_integrator('vode', method = 'bdf', order = 10, nsteps = 30000)
    #ode.set_initial_value(numpy.zeros(N), 0.0)
    #ode.set_f_params(N, D, b, dx)
    #ode.set_jac_params(N, D, b, dx)
    #y = ode.integrate(T)

reload(diffusion_f)

dx = xs[1] - xs[0]
T = 1.0
N = len(at)

ttime = 0.0
tmp = time.time()
dt = 0.01


def UgradU(q, debug = False):
    D, b, sigma = q
    #numpy.concatenate([du, ddudD, ddudb])

    t = 0.0

    bv = numpy.zeros(N)
    bv[0] = b

    v = numpy.zeros(N)
    v[0] = dt * D / dx**2

    u = numpy.zeros(N)
    du = numpy.zeros(N)
    dudD = numpy.zeros(N)
    dudb = numpy.zeros(N)
    while t < T:
        if t + dt > T:
            ndt = T - t
        else:
            ndt = dt

        alpha = ndt * D / dx**2

        u = diffusion_f.solve(numpy.ones(N) * (1 + 2 * alpha), -alpha, u + alpha * bv)

        du = diffusion_f.func(t, u, N, D, b, dx)

        dudD = diffusion_f.solve(numpy.ones(N) * (1 + 2 * alpha), -alpha, dudD + dt * du / D)
        dudb = diffusion_f.solve(numpy.ones(N) * (1 + 2 * alpha), -alpha, dudb + v)

        t += ndt

    logp = sum(0.5 * (-((u - at) **2 / sigma**2) + numpy.log(1.0 / (2 * numpy.pi)) - 2 * numpy.log(sigma)))

    dlpdu = (at - u) / sigma ** 2
    dlpdsigma = sum((-sigma ** 2 + (u - at) **2) / sigma ** 3)

    dlpdD = dlpdu.dot(dudD)
    dlpdb = dlpdu.dot(dudb)

    if not debug:
        return logp, numpy.array([dlpdD, dlpdb, dlpdsigma])
    else:
        return logp, numpy.array([dlpdD, dlpdb, dlpdsigma]), u, dudD, dudb

logp, _, u1, dudD, dudB = UgradU([0.1, max(at), 0.25], True)
logp, _, u2, dudD, dudB = UgradU([0.1, max(at) + 0.01, 0.25], True)

for a, b in zip((u2 - u1) / (0.01), dudB):
    print a / b

print ttime, time.time() - tmp

plt.plot(xs, u)
plt.plot(xs, at)
plt.legend(['Computed', 'Measured'])
plt.show()
#%%

model = """
functions {
    real[] sho(real t,
        real[] u,
        real[] theta,
        real[] x_r,
        int[] x_i) {
            real f[x_i[1]];

            {
                int N;

                N <- x_i[1];

                f[1] <- theta[1] * (x_r[2] - 2 * u[1] + u[2]) / (x_r[1] * x_r[1]);

                for (i in 2 : N - 1)
                    f[i] <- theta[1] * (u[i - 1] - 2 * u[i] + u[i + 1]) / (x_r[1] * x_r[1]);

                f[N] <- theta[1] * (u[N - 1] - 2 * u[N]) / (x_r[1] * x_r[1]);
            }

            return f;
    }
}

data {
    int<lower=1> N;
    real u_measured[N];
    real dx;
    real b; // Left boundary concentration
    real T[1];
}

transformed data {
    real x_r[2];
    int x_i[1];

    x_i[1] <- N;
    x_r[1] <- dx;
    x_r[2] <- b;
}

parameters {
    vector<lower=0>[2] sigma;
    real theta[1];
}

model {
    real u_hat[1, N];
    real u0[N];

    sigma ~ cauchy(0, 2.5);
    theta ~ normal(0, 10);

    for (n in 1:N)
        u0[n] <- 0.0;

    u_hat <- integrate_ode(sho, u0, 0.0, T, theta, x_r, x_i);

    for (n in 1:N)
        u_measured[n] ~ normal(u_hat[n], sigma);
}"""

sm = pystan.StanModel(model_code = model)
#%%

fit = sm.sampling(data = {
    'u_measured' : at,
    'dx' : dx,
    'T' : [T],
    'b' : b,
    'N' : N
})

#%%