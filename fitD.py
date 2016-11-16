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

N = len(at) - 1
x = xs.copy()
dx = x[1] - x[0]

A = numpy.eye(N, N, k = -1) - 2 * numpy.eye(N, N) + numpy.eye(N, N, k = 1)

b = numpy.zeros(N)

b[0] = 1.0
b[N - 1] = 0.0

dt = 0.0001
T = 0.05

def solve(D):
    u = numpy.zeros(N)
    g = 0
    dudD = numpy.zeros(N)

    t = 0.0

    while t < T:
        u = u + dt * D * (A.dot(u) + b) / dx**2

        dudD = dudD + dt * ((A.dot(u) + b) / dx**2 + D * (A.dot(dudD)) / dx**2)

        t += dt

    return u, dudD

data = at[1:].copy()
#%%
dt = 0.0001
print solve(1.0)
#%%

N = 10

X = 1.0
dx = X / N
x = numpy.linspace(0.0, X, N + 2)

A = numpy.eye(N, N, k = -1) - 2 * numpy.eye(N, N) + numpy.eye(N, N, k = 1)

b = numpy.zeros(10)

b[0] = 1.0

dt = 0.001
T = 0.05

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

    return u, dudD

data, g = solve(1.0)

plt.plot(u1, '*')

data += numpy.random.randn(len(u1)) * 0.01

plt.plot(u1, 'r*')

#%%

def loss(D):
    u, dudD = solve(D)

    l = sum((u - data)**2)
    dldD = 2 * (u - data).dot(dudD)

    return l, dldD

D = 2.5

llast = None
while 1:
    l, dldD = loss(D)

    if llast is not None and numpy.abs(l - llast) / l < 1e-7:
        break

    llast = l

    D -= dldD * 0.01

    print "Loss {0}, D = {1}".format(l, D)

#%%
plt.plot(x[1:], solve(D)[0], '*')
plt.plot(x[1:], data, 'r*')
plt.legend(['Solution', 'Measured'])
plt.show()

#%%
u2, g = solve(1.0001)

print (u2 - u1) / (1.0001 - 1.0)
print g
print u1