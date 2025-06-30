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
  real<lower=0> phi;
}

model {
  vector[N] log_lambda;

  alpha ~ normal(2, 1.5);
  beta_cont ~ student_t(10, 0, 0.5);
  beta_weaptype ~ student_t(10, 0, 0.5);
  beta_targtype ~ student_t(10, 0, 0.5);
  beta_country ~ student_t(10, 0, 0.5);
  phi ~ exponential(1);

  for (n in 1:N) {
    log_lambda[n] = alpha + 
                    dot_product(row(X_cont, n), beta_cont) +
                    beta_weaptype[weaptype[n]] +
                    beta_targtype[targtype[n]] +
                    beta_country[country[n]];
  }

  nkill ~ neg_binomial_2_log(log_lambda, phi);
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
    nkill_pred[n] = neg_binomial_2_log_rng(log_lambda_n, phi);
    log_lik[n] = neg_binomial_2_log_lpmf(nkill[n] | log_lambda_n, phi);
  }
}