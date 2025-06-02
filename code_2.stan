data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=0> nkill;
}

parameters {
  real alpha;
  vector[K] beta;
  real<lower=0> phi;  // parametr rozproszenia (overdispersion)
}

model {
  alpha ~ normal(0.6647, 0.83); 
  beta ~ normal(0, 0.5);
  phi ~ exponential(1);  // prior dla phi (możesz zmienić, np. half-normal)

  nkill ~ neg_binomial_2_log(alpha + X * beta, phi);
}

generated quantities {
  array[N] int nkill_pred;
  for (n in 1:N) {
    nkill_pred[n] = neg_binomial_2_log_rng(alpha + dot_product(row(X, n), beta), phi);
  }
}
