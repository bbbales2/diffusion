functions {
  real[] sho(real t, real[] w, real[] p, real[] x_r, int[] x_i) {
    int N = size(w);
    real dwdt[N];
    real D = p[1];
    real alpha = p[2];
    real dx = x_r[1];
    real w0 = x_r[2];
    real wNp1 = x_r[3];
    real u = alpha * sqrt(D / (t + 0.1));
    
    dwdt[1] = (D * (w[2] - w[1]) - D * (w[1] - w0)) / dx^2 + u * (w[2] - w0) / (2 * dx);
    dwdt[N] = (D * (wNp1 - w[N]) - D * (w[N] - w[N - 1])) / dx^2 + u * (wNp1 - w[N - 1]) / (2 * dx);
    
    for(i in 2 : (N - 1)) {
      dwdt[i] = (D * (w[i + 1] - w[i]) - D * (w[i] - w[i - 1])) / dx^2 + u * (w[i + 1] - w[i - 1]) / (2 * dx);
    }
    
    return dwdt;
  }
}

data {
  int<lower=1> N;
  real t;
  real w0;
  real wNp1;
  real w_init[N];
  real dx;
  vector[N] y;
}

transformed data {
  real x_r[3] = { dx, w0, wNp1 };
  real ts[1] = { t };
  int x_i[0];
}

parameters {
  real<lower=0> D;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

transformed parameters {
  real wh[1, N];
  real p[2] = { D, alpha };
  
  wh = integrate_ode_rk45(sho, w_init, 0.0, ts, p, x_r, x_i,
                          1e-6, 1e-6, 100);
}

model {
  D ~ normal(0.1, 0.1) T[0.0,];
  alpha ~ normal(0.1, 0.5) T[0.0,];
  sigma ~ normal(0, 1.0) T[0.0,];
  
  y ~ normal(wh[1, :], sigma);
}
