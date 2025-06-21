data {
  int<lower=0> N;                                    // number of observations
  int<lower=0> K_cont;                               // number of continuous predictors
  int<lower=1> K_weaptype;                           // number of weapon types
  int<lower=1> K_targtype;                           // number of target types
  int<lower=1> K_country;                            // number of countries
  matrix[N, K_cont] X_cont;                          // continuous predictors (nperps)
  array[N] int<lower=1, upper=K_weaptype> weaptype;  // weapon type (1-indexed)
  array[N] int<lower=1, upper=K_targtype> targtype;  // target type (1-indexed)
  array[N] int<lower=1, upper=K_country> country;    // country (1-indexed)
}

generated quantities {
  real alpha = normal_rng(3.5, 1.25);
  vector[K_cont] beta_cont;
  vector[K_weaptype] beta_weaptype;
  vector[K_targtype] beta_targtype;
  vector[K_country] beta_country;

  array[N] int nkill_prior_pred;
  vector[N] log_lambda_prior;

  for (k in 1:K_cont) {
    beta_cont[k] = normal_rng(0.05, 0.25);
  }
  for (k in 1:K_weaptype) {
    beta_weaptype[k] = normal_rng(0.5, 0.5);
  }
  for (k in 1:K_targtype) {
    beta_targtype[k] = normal_rng(0.15, 0.5);
  }
  for (k in 1:K_country) {
    beta_country[k] = normal_rng(0.15, 0.5);
  }

  for (n in 1:N) {
    log_lambda_prior[n] = alpha + 
                          dot_product(row(X_cont, n), beta_cont) + 
                          beta_weaptype[weaptype[n]] + 
                          beta_targtype[targtype[n]] + 
                          beta_country[country[n]];

    nkill_prior_pred[n] = poisson_log_rng(log_lambda_prior[n]);
  }
}
