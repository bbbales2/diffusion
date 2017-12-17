functions {
  real[] sho(real t, real[] u, real[] p, real[] dx, int[] x_i) {
    int N = size(u);
    real dudt[N];
    real D = p[1];
    
    dudt[1] = (D * (u[2] - u[1]) - D * (u[1] - 1.0)) / dx[1]^2;
    dudt[N] = (D * (0.0 - u[N]) - D * (u[N] - u[N - 1])) / dx[1]^2;
    
    for(i in 2 : (N - 1)) {
      dudt[i] = (D * (u[i + 1] - u[i]) - D * (u[i] - u[i - 1])) / dx[1]^2;
    }
    
    return dudt;
  }
}

data {
  int<lower=1> N;
  real t;
  real u0[N];
  real dx;
  vector[N] y;
}

transformed data {
  real x_r[1];
  real ts[1];
  int x_i[0];
  
  ts[1] = t;
  x_r[1] = dx;
}

parameters {
  real<lower=0> D;
  real<lower=0> sigma;
}

transformed parameters {
  real uh[1, N];
  real p[1] = { D };
  
  uh = integrate_ode_rk45(sho, u0, 0.0, ts, p, x_r, x_i,
          1e-6, 1e-6, 100);
}

model {
  D ~ normal(0.1, 0.1);
  sigma ~ normal(0, 1.0);
  
  y ~ normal(uh[1, :], sigma);
}
