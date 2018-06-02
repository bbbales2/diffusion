library(tidyverse)
library(ggplot)
library(rstan)

cinf = 0.0
Cc = 93.5 / 7.82
tf = 60.0
dt = 0.25
M = as.integer(tf / dt)
dx = 10.0
S = as.integer(max(df$CorrDist) / dx) + 10

df = read_csv("datafitting-master/data/PX7YSZ/Kramer/1350/Sample24_3min.csv") %>%
  select(CorrDist, ZrO2) %>%
  drop_na() %>%
  arrange(CorrDist) %>%
  mutate(ZrO2 = ZrO2 / 7.82)

fit = stan("models/collin.stan",
           data = list(N = nrow(df),
                       y = df$ZrO2,
                       x = df$CorrDist,
                       cinf = cinf,
                       Cc = Cc,
                       M = M,
                       dt = dt,
                       dx = dx,
                       S = S),
           iter = 2000, chains = 4, cores = 4)

extract(fit, "yhat")$yhat %>%
  as.tibble %>%
  setNames(((1:S) - 1) * dx) %>%
  mutate(rn = row_number()) %>%
  gather(x, y, -rn) %>%
  mutate(x = as.numeric(x)) %>%
  group_by(x) %>%
  summarize(q1 = quantile(y, 0.1),
            q2 = quantile(y, 0.9)) %>%
  ggplot(aes(x)) +
  geom_ribbon(aes(ymin = q1, ymax = q2), alpha = 0.5) +
  geom_point(data = df, aes(CorrDist, ZrO2), color = "red")

extract(fit, c("D", "u0")) %>%
  as.tibble %>%
  ggpairs(lower = list(continuous = wrap("points", alpha = 0.2)))
