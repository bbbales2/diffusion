library(tidyverse)
library(ggplot)
library(rstan)

N = 100
cinf = 0.0
Cc = 10
y = rep(0, N)
x = seq(0, 300, length = N)
tf = 60.0
dt = 0.5
M = as.integer(tf / dt)

fit = stan("models/collin.stan",
           data = list(N = N,
                       y = y,
                       x = x,
                       cinf = cinf,
                       Cc = Cc,
                       M = M,
                       dt = dt),
           init = list(list(D = 4.0, u0 = 0.1)), iter = 1000, chains = 1, cores = 1)
