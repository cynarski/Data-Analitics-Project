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
  
  alpha ~ normal(2, 1.5);
  beta_cont ~ normal(0, 0.5);
  beta_weaptype ~ normal(0, 0.5);
  beta_targtype ~ normal(0, 0.5);
  beta_country ~ normal(0, 0.5);
  
  for (n in 1:N) {
    real zi_prob = 0.5;  // fixed zi_prob like in your prior predictive
    log_lambda[n] = alpha + 
                    dot_product(row(X_cont, n), beta_cont) +
                    beta_weaptype[weaptype[n]] +
                    beta_targtype[targtype[n]] +
                    beta_country[country[n]];
    
    if (log_lambda[n] > 20.794415) {
      log_lambda[n] = 20;
    }
    
    if (nkill[n] == 0) {
      target += log_sum_exp(log(zi_prob),
                           log(1 - zi_prob) + poisson_log_lpmf(0 | log_lambda[n]));
    } else {
      target += log(1 - zi_prob) + poisson_log_lpmf(nkill[n] | log_lambda[n]);
    }
  }
}

generated quantities {
  array[N] int nkill_pred;
  vector[N] log_lik;
  array[N] int is_zero;
  
  for (n in 1:N) {
    real zi_prob = 0.5;
    real log_lambda_n = alpha + 
                        dot_product(row(X_cont, n), beta_cont) +
                        beta_weaptype[weaptype[n]] +
                        beta_targtype[targtype[n]] +
                        beta_country[country[n]];
    
    if (log_lambda_n > 20.794415) {
      log_lambda_n = 20;
    }
    
    is_zero[n] = bernoulli_rng(zi_prob);
    if (is_zero[n] == 1) {
      nkill_pred[n] = 0;
    } else {
      nkill_pred[n] = poisson_log_rng(log_lambda_n);
    }
    
    if (nkill[n] == 0) {
      log_lik[n] = log_sum_exp(log(zi_prob),
                              log(1 - zi_prob) + poisson_log_lpmf(0 | log_lambda_n));
    } else {
      log_lik[n] = log(1 - zi_prob) + poisson_log_lpmf(nkill[n] | log_lambda_n);
    }
  }
}













/*
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

  alpha ~ normal(-5, 3.25);
  beta_cont ~ normal(0, 0.15);
  beta_weaptype ~ normal(0, 0.5);
  beta_targtype ~ normal(0, 0.5);
  beta_country ~ normal(0, 0.5);


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
}*/