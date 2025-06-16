data {
  int<lower=0> N;          
  int<lower=0> K;         
  matrix[N, K] X;       
}

generated quantities {
  real alpha;
  vector[K] beta;
  array[N] int nkill_sim;

  alpha = normal_rng(2.5, 0.75);
  for (k in 1:K)
    beta[k] = normal_rng(0, 0.25);

  for (n in 1:N) {
    nkill_sim[n] = poisson_log_rng(alpha + dot_product(row(X, n), beta));
  }
}
