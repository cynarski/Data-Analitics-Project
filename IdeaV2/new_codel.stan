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
  real alpha;                                        // intercept
  vector[K_cont] beta_cont;                          // continuous predictor coefficients (nperps)
  vector[K_weaptype] beta_weaptype;                  // weapon type effects
  vector[K_targtype] beta_targtype;                  // target type effects
  vector[K_country] beta_country;                    // country effects
}

model {
  vector[N] log_lambda;
  
  // Priors
  alpha ~ normal(2.5, 0.75);
  beta_cont ~ normal(0, 0.25);
  beta_weaptype ~ normal(0, 0.25);
  beta_targtype ~ normal(0, 0.25);
  beta_country ~ normal(0, 0.25);
  
  // Linear predictor
  for (n in 1:N) {
    log_lambda[n] = alpha + 
                    dot_product(row(X_cont, n), beta_cont) +
                    beta_weaptype[weaptype[n]] +
                    beta_targtype[targtype[n]] +
                    beta_country[country[n]];
  }
  
  // Likelihood
  nkill ~ poisson_log(log_lambda);
}

generated quantities {
  array[N] int nkill_pred;
  vector[N] log_lik;                                 // log-likelihood for LOO/WAIC
  
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