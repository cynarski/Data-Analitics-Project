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
  real phi = exponential_rng(1);
  array[N] int nkill_prior_pred;
  vector[N] log_lambda_prior;
  
  for (k in 1:K_cont) {
    beta_cont[k] = student_t_rng(10, 0, 0.5);
  }
  for (k in 1:K_weaptype) {
    beta_weaptype[k] = student_t_rng(10, 0, 0.5);
  }
  for (k in 1:K_targtype) {
    beta_targtype[k] = student_t_rng(10, 0, 0.5);
  }
  for (k in 1:K_country) {
    beta_country[k] = student_t_rng(10, 0, 0.5);
  }


    
  for (n in 1:N) {
    log_lambda_prior[n] = alpha + 
                          dot_product(row(X_cont, n), beta_cont) + 
                          beta_weaptype[weaptype[n]] + 
                          beta_targtype[targtype[n]] + 
                          beta_country[country[n]];
    if (log_lambda_prior[n] > 10) {
        log_lambda_prior[n] = 10;
    } else if (log_lambda_prior[n] < -10) {
        log_lambda_prior[n] = -10;
    }
    if (phi == 0) {
        phi = 0.001;
    }

    nkill_prior_pred[n] = neg_binomial_2_log_rng(log_lambda_prior[n], phi);
  }
}