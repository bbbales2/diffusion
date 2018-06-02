functions {
  real[] f(real t, real[] c, real[] theta, real[] x_r, int[] x_i) {
    real D = theta[1];
    real u0 = theta[2];
    real dx = x_r[1];
    real cinf = x_r[2];
    real Cc = x_r[3];
    int N = size(c);
    real u = u0 * (1 - c[1]);
    real dcdx0 = u * (c[1] - Cc) / D;
    real f[N];
    
    f[1] = D * ((c[2] - c[1]) / dx - dcdx0) / (0.5 * dx) - u * dcdx0;
    for (i in 2:(N - 1)) {
      f[i] = D * (c[i + 1] - 2 * c[i] + c[i - 1]) / (dx * dx) - u * (c[i + 1] - c[i - 1]) / (2 * dx);
    }
    f[N] = D * (cinf - 2 * c[N] + c[N - 1]) / (dx * dx) - u * (cinf - c[N - 1]) / (2 * dx);

    return f;
  }
}

data {
  int N;
  real x[N];
  real y[N];
  real cinf;
  real Cc;
  int M;
  real dt;
  real dx;
  int S;
}

transformed data {
  real y0[S] = rep_array(0.0, S);
  real x_r[3] = { dx, cinf, Cc };
  int x_i[0];
  int j = 1;
  real alpha[N];
  int ode_j[N];
  real x_ode[S];

  for(i in 1:S) {
    x_ode[i] = (i - 1) * dx;
  }

  for(i in 1:N) {
    real total;
    
    while(x_ode[j] < x[i]) {
      j = j + 1;
    }
    
    ode_j[i] = j;
    total = x_ode[j] - x_ode[j - 1];
    alpha[i] = (x_ode[j] - x[i]) / total;
  }
}

parameters {
  real<lower = 0.0> D;
  real<lower = 0.0> u0;
  real<lower = 0.0> sigma;
}

transformed parameters {
  real theta[2] = { D, u0 };
  real ymean[S];
  {
    vector[S] y_ = to_vector(y0);
    for(i in 1:M) {
      y_ = y_ + dt * to_vector(f(0.0, to_array_1d(y_), theta, x_r, x_i));
    }
    ymean = to_array_1d(y_);
  }
  //real yhat[N] = integrate_ode_rk45(f, y0, 0.0, { tf }, theta, x_r, x_i, 1e-5, 1e-5, 200)[1,];
}

model {
  D ~ normal(26.0, 1.0);
  //u0 ~ normal(0.1, 0.1);
  
  for(i in 1:N) {
    int jj = ode_j[i];
    real yinterp = (1 - alpha[jj]) * ymean[jj] + alpha[jj] * ymean[jj - 1];
    
    y[i] ~ normal(yinterp, sigma);
  }
}

generated quantities {
  real yhat[S];
  
  for(i in 1:S) {
    yhat[i] = normal_rng(ymean[i], sigma);
  }
}
