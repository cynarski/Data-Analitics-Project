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
  real alpha = normal_rng(2, 1.5);
  vector[K_cont] beta_cont;
  vector[K_weaptype] beta_weaptype;
  vector[K_targtype] beta_targtype;
  vector[K_country] beta_country;
  array[N] int is_zero;
  
  array[N] int nkill_prior_pred;
  vector[N] log_lambda_prior;
  
  for (k in 1:K_cont) {
    beta_cont[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_weaptype) {
    beta_weaptype[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_targtype) {
    beta_targtype[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_country) {
    beta_country[k] = normal_rng(0, 0.5);
  }

  real zi_prob = beta_rng(1, 1);

  for (n in 1:N) {
    

    log_lambda_prior[n] = alpha + 
                          dot_product(row(X_cont, n), beta_cont) + 
                          beta_weaptype[weaptype[n]] + 
                          beta_targtype[targtype[n]] + 
                          beta_country[country[n]];
    if (log_lambda_prior[n] > 20.794415) {
        log_lambda_prior[n] = 20;
    }

    is_zero[n] = bernoulli_rng(zi_prob);
    if (is_zero[n] == 0) {
      nkill_prior_pred[n] = 0;
    } 
    else {
      nkill_prior_pred[n] = poisson_log_rng(log_lambda_prior[n]);
    }
  }
}






/*
generated quantities {
  // Prior samples for Poisson component
  real alpha = normal_rng(-1, 0.5);
  vector[K_cont] beta_cont;
  vector[K_weaptype] beta_weaptype;
  vector[K_targtype] beta_targtype;
  vector[K_country] beta_country;
  
  // Prior samples for zero-inflation component
  real gamma = normal_rng(-4.5, 3.25);  // zero-inflation intercept
  vector[K_cont] delta_cont;
  vector[K_weaptype] delta_weaptype;
  vector[K_targtype] delta_targtype;
  vector[K_country] delta_country;
  
  // Output variables
  array[N] int nkill_prior_pred;
  vector[N] log_lambda_prior;
  vector[N] zi_prob_prior;
  
  // Summary statistics for diagnostics
  real mean_nkill_prior_pred;
  real prop_zero_prior_pred;
  real max_nkill_prior_pred;
  
  // Sample coefficients
  for (k in 1:K_cont) {
    beta_cont[k] = normal_rng(0, 0.25);
    delta_cont[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_weaptype) {
    beta_weaptype[k] = normal_rng(0, 0.5);
    delta_weaptype[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_targtype) {
    beta_targtype[k] = normal_rng(0, 0.5);
    delta_targtype[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_country) {
    beta_country[k] = normal_rng(0, 0.5);
    delta_country[k] = normal_rng(0, 0.5);
  }
  
  // Generate predictions
  for (n in 1:N) {
    // Poisson rate (log scale)
    log_lambda_prior[n] = alpha + 
                          dot_product(row(X_cont, n), beta_cont) + 
                          beta_weaptype[weaptype[n]] + 
                          beta_targtype[targtype[n]] + 
                          beta_country[country[n]];
    
    // Zero-inflation probability (logit scale)
    real logit_zi = gamma +
                    dot_product(row(X_cont, n), delta_cont) +
                    delta_weaptype[weaptype[n]] +
                    delta_targtype[targtype[n]] +
                    delta_country[country[n]];
    
    zi_prob_prior[n] = inv_logit(logit_zi);
    
    // Numerical stability: cap log_lambda at reasonable value
    if (log_lambda_prior[n] > 15) {
        log_lambda_prior[n] = 15;  // exp(15) ≈ 3.3 million
    }
    if (log_lambda_prior[n] < -10) {
        log_lambda_prior[n] = -10;  // exp(-10) ≈ 0.000045
    }
    
    // Generate from zero-inflated Poisson
    if (bernoulli_rng(zi_prob_prior[n])) {
      nkill_prior_pred[n] = 0;  // structural zero
    } else {
      nkill_prior_pred[n] = poisson_log_rng(log_lambda_prior[n]);
    }
  }
}
*/







/*
generated quantities {
  real alpha = normal_rng(-4.5, 3.25); //-5, 3.25
  vector[K_cont] beta_cont;
  vector[K_weaptype] beta_weaptype;
  vector[K_targtype] beta_targtype;
  vector[K_country] beta_country;

  array[N] int nkill_prior_pred;
  vector[N] log_lambda_prior;

  for (k in 1:K_cont) {
    beta_cont[k] = normal_rng(0, 0.25);
  }
  for (k in 1:K_weaptype) {
    beta_weaptype[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_targtype) {
    beta_targtype[k] = normal_rng(0, 0.5);
  }
  for (k in 1:K_country) {
    beta_country[k] = normal_rng(0, 0.5);
  }

  for (n in 1:N) {
    log_lambda_prior[n] = alpha + 
                          dot_product(row(X_cont, n), beta_cont) + 
                          beta_weaptype[weaptype[n]] + 
                          beta_targtype[targtype[n]] + 
                          beta_country[country[n]];
    if (log_lambda_prior[n] > 20.794415) {
        log_lambda_prior[n] = 20;
    }

    nkill_prior_pred[n] = poisson_log_rng(log_lambda_prior[n]);
  }
}
*/