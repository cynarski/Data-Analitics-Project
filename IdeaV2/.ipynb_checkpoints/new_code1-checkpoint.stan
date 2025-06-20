data {
  int<lower=0> N;                                    // number of observations
  int<lower=0> K_cont;                               // number of continuous predictors (1 = nperps)
  int<lower=1> K_weaptype;                           // number of weapon types
  int<lower=1> K_targtype;                           // number of target types
  int<lower=1> K_country;                            // number of countries
  matrix[N, K_cont] X_cont;                          // continuous predictors (nperps)
  array[N] int<lower=1, upper=K_weaptype> weaptype;  // weapon type (1-indexed)
  array[N] int<lower=1, upper=K_targtype> targtype;  // target type (1-indexed)
  array[N] int<lower=1, upper=K_country> country;    // country (1-indexed)
  array[N] int<lower=0> nkill;                       // outcome variable
}

parameters {
  real alpha;
  vector[K_cont] beta_cont;
  vector[K_weaptype] beta_weaptype;
  vector[K_targtype] beta_targtype;
  vector[K_country] beta_country;
}

model {
  vector[N] log_lambda;
  
  alpha ~ normal(1.5, 0.75);
  beta_cont ~ normal(0, 0.5);
  beta_weaptype ~ normal(0, 0.5);
  beta_targtype ~ normal(0, 0.5);
  beta_country ~ normal(0, 0.5);

/*
  alpha ~ normal(3.5, 1.25);
  beta_cont ~ normal(0.05, 0.25);
  beta_weaptype ~ normal(0.5, 0.5);
  beta_targtype ~ normal(0.15, 0.5);
  beta_country ~ normal(0.15, 0.5);
*/

  for (n in 1:N) {
    log_lambda[n] = alpha + 
                    dot_product(row(X_cont, n), beta_cont) +
                    beta_weaptype[weaptype[n]] +
                    beta_targtype[targtype[n]] +
                    beta_country[country[n]];
  }
  
  nkill ~ poisson_log(log_lambda);
}

generated quantities {
  array[N] int nkill_pred;
  vector[N] log_lik;
  
  for (n in 1:N) {
    real log_lambda_n = alpha + 
                        dot_product(row(X_cont, n), beta_cont) +
                        beta_weaptype[weaptype[n]] +
                        beta_targtype[targtype[n]] +
                        beta_country[country[n]];
    nkill_pred[n] = poisson_log_rng(log_lambda_n);
    log_lik[n] = poisson_log_lpmf(nkill[n] | log_lambda_n);
  }
}