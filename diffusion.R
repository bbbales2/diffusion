library(MASS)
options(mc.cores = parallel::detectCores())
library(ggplot2)
library(deSolve)
library(GGally)
library(purrr)
library(parallel)
library(ggthemes)
library(tidyverse)
library(rstan)
library(shinystan)
library(bayesplot)

N = 10

x = seq(0.0, 1.0, length = N + 2)
dx = x[[2]] - x[[1]]
D = 0.1

func = function(t, u, D) {
  dudt = rep(0, N)
  dudt[[1]] = (D * (u[[2]] - u[[1]]) - D * (u[[1]] - 1.0)) / dx^2
  dudt[[N]] = (D * (0.0 - u[[N]]) - D * (u[[N]] - u[[N - 1]])) / dx^2
  for(i in 2 : (N - 1)) {
    dudt[[i]] = (D * (u[[i + 1]] - u[[i]]) - D * (u[[i]] - u[[i - 1]])) / dx^2
  }
  list(dudt)
}

h = 0.01
times = seq(0, 0.1, by = h)

u0 = rep(0, N)
fout = inner_join(as_tibble(ode(y = u0, times = times, func = func, parms = D)[,]) %>%
                    gather(xi, u, 2:(N + 1)) %>%
                    mutate(xi = as.integer(xi)) %>%
                    mutate(unoise = u + rnorm(n(), sd = 0.01)),
                  as_tibble(list(xi = 1:N, x = x[2:(N + 1)])),
                  by = "xi"
)

fout %>% filter(time == 0.1) %>%
  gather(name, u, c(u, unoise)) %>%
  ggplot(aes(x, u)) +
  geom_line(aes(group = name, color = name))

fit = stan("models/diffusion.stan", data = list(N = N,
                                                t = 0.1,
                                                u0 = u0,
                                                y = (fout %>% filter(time == 0.1))$unoise,
                                                dx = dx), chains = 4, iter = 1000)

print(fit)

mcmc_trace(extract(fit, c("D"), permute = FALSE))
mcmc_pairs(extract(fit, c("D", "sigma"), permute = FALSE))
mcmc_combo(extract(fit, c("D", "sigma"), permute = FALSE))

launch_shinystan(fit)
