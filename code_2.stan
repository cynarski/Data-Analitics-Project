data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=0> nkill;
}

parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> phi; 
}

model {
  alpha ~ normal(1.5, 0.75); 
  beta ~ normal(0, 0.5);
  phi ~ exponential(0.5);

  nkill ~ neg_binomial_2_log(alpha + X * beta, phi);
}

generated quantities {
  array[N] int nkill_pred;
  for (n in 1:N) {
    nkill_pred[n] = neg_binomial_2_log_rng(alpha + dot_product(row(X, n), beta), phi);
  }
}