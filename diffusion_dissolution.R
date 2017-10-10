library(MASS)
library(ggplot2)
library(deSolve)
library(tidyverse)
library(rstan)
library(shinystan)
library(bayesplot)

N = 10

x = seq(0.0, 1.0, length = N + 2)
dx = x[[2]] - x[[1]]
D = 0.1
alpha = 1.0
T = 0.1
w0 = 1.0
wNp1 = 0.0

func = function(t, w, parms) {
  D = parms[[1]]
  alpha = parms[[2]]
  u = alpha * sqrt(D / (0.1 + t))
  dudt = rep(0, N)
  dudt[[1]] = (D * (w[[2]] - w[[1]]) - D * (w[[1]] - 1.0)) / dx^2 + u * (w[[2]] - w0) / (2 * dx)
  dudt[[N]] = (D * (wNp1 - w[[N]]) - D * (w[[N]] - w[[N - 1]])) / dx^2 + u * (wNp1 - w[[N - 1]]) / (2 * dx)
  for(i in 2 : (N - 1)) {
    dudt[[i]] = (D * (w[[i + 1]] - w[[i]]) - D * (w[[i]] - w[[i - 1]])) / dx^2 + u * (w[[i + 1]] - w[[i - 1]]) / (2 * dx)
  }
  list(dudt)
}

h = 0.01
times = seq(0, T, by = h)

w_init = rep(0, N)
(fout = inner_join(as_tibble(ode(y = w_init, times = times, func = func, parms = c(D, alpha))[,]) %>%
                    gather(xi, w, 2:(N + 1)) %>%
                    mutate(xi = as.integer(xi)) %>%
                    mutate(unoise = w + rnorm(n(), sd = 0.01)),
                  as_tibble(list(xi = 1:N, x = x[2:(N + 1)])),
                  by = "xi"
)) %>% filter(time == T) %>%
  gather(name, w, c(w, unoise)) %>%
  ggplot(aes(x, w)) +
  geom_line(aes(group = name, color = name))

fit = stan("models/diffusion_disolution.stan",
           data = list(N = N,
                       t = 0.1,
                       w0 = w0,
                       wNp1 = wNp1,
                       w_init = w_init,
                       y = (fout %>% filter(time == 0.1))$unoise,
                       dx = dx),
           chains = 1,
           cores = 4,
           iter = 1000)

print(fit)

mcmc_trace(extract(fit, c("D"), permute = FALSE))
mcmc_pairs(extract(fit, c("D", "alpha", "sigma"), permute = FALSE))
mcmc_combo(extract(fit, c("D", "alpha", "sigma"), permute = FALSE))

launch_shinystan(fit)
