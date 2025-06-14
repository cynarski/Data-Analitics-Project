data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=0> nkill;
}

parameters {
  real alpha;
  vector[K] beta;
}

model {
  alpha ~ normal(2.5, 0.75); 
  beta ~ normal(0, 0.25);

  nkill ~ poisson_log(alpha + X * beta);
}

generated quantities {
  array[N] int nkill_pred;
  for (n in 1:N) {
    nkill_pred[n] = poisson_log_rng(alpha + dot_product(row(X, n), beta));
  }
}
