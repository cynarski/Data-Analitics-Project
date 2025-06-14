data {
  int<lower=0> N;          
  int<lower=0> K;         
  matrix[N, K] X;       
}

generated quantities {
  real alpha;               
  vector[K] beta;           
  real<lower=0> phi;       
  array[N] int nkill_sim; 

  // Priory
  alpha = normal_rng(1.5, 0.75);      
  for (k in 1:K)
    beta[k] = normal_rng(0, 0.5); 
  phi = exponential_rng(0.5); 

  // Symulacja danych
  for (n in 1:N) {
    nkill_sim[n] = neg_binomial_2_log_rng(alpha + dot_product(row(X, n), beta), phi);
  }
}
