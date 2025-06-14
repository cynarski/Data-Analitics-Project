data {
  int<lower=0> N;          
  int<lower=0> K;         
  matrix[N, K] X;       
}

generated quantities {
  real alpha;               // intercept z priory
  vector[K] beta;           // współczynniki z priory
  array[N] int nkill_sim;   // wygenerowana liczba ofiar/incydentów

  // Generowanie priors
  alpha = normal_rng(2.5, 0.75);       // większe prawdopodobieństwo λ ≈ 2–10
  for (k in 1:K)
    beta[k] = normal_rng(0, 0.25);        // delikatne efekty predyktorów

  for (n in 1:N) {

    nkill_sim[n] = poisson_log_rng(alpha + dot_product(row(X, n), beta));
  }
}
