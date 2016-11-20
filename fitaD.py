#%%

import numpy
import matplotlib.pyplot as plt
import scipy

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

xs = xs[1:]
at = at[1:]

#%%

N = len(xs)
#xs = numpy.linspace(0, 1, N + 2)
dx = xs[1] - xs[0]

D = None
a = 1.0

b1 = 1.0
b2 = 0.0

#for a in [-0.0001, 0.0, 0.0001]:
def f(y, t0, D, a):
    dcdt = numpy.zeros(N)
    dpdt = numpy.zeros(N)
    dqdt = numpy.zeros(N)

    co = y[:N]
    po = y[N:2*N]
    qo = y[2*N:]

    f = (D * (co[1] - 2 * co[0] + b1) / dx**2 + D * (a / 2.0) * (co[1]**2 - 2 * co[0]**2 + b1**2) / dx**2)

    dcdt[0] = f

    fi = -2 * D / dx**2 + -2 * a * D * co[0] / dx**2
    fip1 = D / dx**2 + a * D * co[1] / dx**2

    dpdt[0] = fi * po[0] + fip1 * po[1] + f / D

    dqdt[0] = fi * qo[0] + fip1 * qo[1] + D * (1 / 2.0) * (co[1]**2 - 2 * co[0]**2 + b1**2) / dx**2

    for i in range(1, N - 1):
        f = (D * (co[i + 1] - 2 * co[i] + co[i - 1]) / dx**2 + D * (a / 2.0) * (co[i + 1]**2 - 2 * co[i]**2 + co[i - 1]**2) / dx**2)

        dcdt[i] = f

        fim1 = D / dx**2 + a * D * co[i - 1] / dx**2
        fi = -2 * D / dx**2 + -2 * a * D * co[i] / dx**2
        fip1 = D / dx**2 + a * D * co[i + 1] / dx**2

        dpdt[i] = fim1 * po[i - 1] + fi * po[i] + fip1 * po[i + 1] + f / D

        dqdt[i] = fim1 * qo[i - 1] + fi * qo[i] + fip1 * qo[i + 1] + D * (1 / 2.0) * (co[i + 1]**2 - 2 * co[i]**2 + co[i - 1]**2) / dx**2

    f = (D * (b2 - 2 * co[N - 1] + co[N - 2]) / dx**2 + D * (a / 2.0) * (b2**2 - 2 * co[N - 1]**2 + co[N - 2]**2) / dx**2)

    dcdt[N - 1] = f

    fim1 = D / dx**2 + a * D * co[N - 2] / dx**2
    fi = -2 * D / dx**2 + -2 * a * D * co[N - 1] / dx**2

    dpdt[N - 1] = fim1 * po[N - 2] + fi * po[N - 1] + f / D

    dqdt[N - 1] = fim1 * qo[N - 2] + fi * qo[N - 1] + D * (1.0 / 2.0) * (b2**2 - 2 * co[N - 1]**2 + co[N - 2]**2) / dx**2

    return numpy.concatenate((dcdt, dpdt, dqdt))

def solve(D, a):
    y = scipy.integrate.odeint(f, numpy.zeros(3 * N), [0.0, 0.1], args = (D, a))

    c = y[-1, : N]
    p = y[-1, N : 2 * N]
    q = y[-1, 2 * N :]

    return c, p, q


#D0 = 1.0
#D1 = D0 * 1.0001
D = 1.0
a0 = 1.00001
a1 = 1.00001 * a0
co0, po, qo = solve(D, a0)
co1, po1, qo1 = solve(D, a1)

approx = (co1 - co0) / (a1 - a0)
plt.plot(approx)
plt.plot(qo)
plt.legend(['Approx', 'Exact'])
plt.show()

#%%

legend = []
for a in [-1.0, 0.0, 1.0]:
    c, p = solve(D, a)
    plt.plot(xs, c)
    legend.append("a = {0}".format(a))
plt.legend(legend)
plt.show()

#%%

def loss(state):
    print state
    D, a = state

    c, p, q = solve(D, a)

    plt.plot(xs, at)
    plt.plot(xs, c)
    plt.legend(['data', 'solution'])
    plt.show()

    return numpy.sum((c - at)**2)#numpy.array([2 * numpy.dot(p, c - at), 2 * numpy.dot(q, c - at)]),

def dloss(state):
    #print state
    D, a = state

    c, p, q = solve(D, a)

    return numpy.array([2 * numpy.dot(p, c - at), 2 * numpy.dot(q, c - at)])

state = numpy.array([1.0, 0.0])
#%%
while 1:
    dlds, l = loss(state)

    print l, dlds, state

    state -= dlds * 0.1
#%%
[ 0.89076687 -0.3387623 ]
#%%
res = scipy.optimize.minimize(loss, [1.0, 1.0], method = 'L-BFGS-B', jac = dloss)#