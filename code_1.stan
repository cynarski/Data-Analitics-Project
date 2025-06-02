data {
  int<lower=0> N;
  int<lower=0> K;
  matrix[N, K] X;
  array[N] int<lower=0> nkill;
}

parameters {
  real alpha;
  vector[K] beta;
}

model {
  alpha ~ normal(0.6647, 0.83); 
  beta ~ normal(0, 0.5);

  nkill ~ poisson_log(alpha + X * beta);
}

generated quantities {
  array[N] int nkill_pred;
  for (n in 1:N) {
    nkill_pred[n] = poisson_log_rng(alpha + dot_product(row(X, n), beta));
  }
}



// data {
//   int<lower=0> N;
//   int<lower=0> K;
//   matrix[N, K] X;
//   array[N] int<lower=0> nkill;
// }

// parameters {
//   real alpha_count;        // Intercept dla count modelu
//   real alpha_zero;         // Intercept dla zero-inflation
//   vector[K] beta_count;    // Coef dla count modelu
//   vector[K] beta_zero;     // Coef dla zero-inflation
//   real<lower=0> phi;       // Rozproszenie NB
// }

// model {
//   // Priory
//     alpha_count ~ normal(0, 1.5);     // Średnia liczba ofiar = exp(alpha)
//     beta_count ~ normal(0, 1);        // Efekt predyktora: x0.37 do x2.7 (95%)
//     alpha_zero ~ normal(0, 1);        // Dla zero-inflation
//     beta_zero ~ normal(0, 1);         // Dla zero-inflation
//     phi ~ exponential(1);

//   for (n in 1:N) {
//     real logit_zi = alpha_zero + dot_product(row(X, n), beta_zero);
//     real mu = exp(alpha_count + dot_product(row(X, n), beta_count));

//     target += log_sum_exp(
//       bernoulli_lpmf(1 | inv_logit(logit_zi)),  // Zero-inflation part
//       bernoulli_lpmf(0 | inv_logit(logit_zi)) + neg_binomial_2_lpmf(nkill[n] | mu, phi)  // Count part
//     );
//   }
// }

// generated quantities {
//   array[N] int nkill_pred;

//   for (n in 1:N) {
//     real logit_zi = alpha_zero + dot_product(row(X, n), beta_zero);
//     real mu = exp(alpha_count + dot_product(row(X, n), beta_count));
//     int is_zero = bernoulli_rng(inv_logit(logit_zi));
//     if (is_zero == 1) {
//       nkill_pred[n] = 0;
//     } else {
//       nkill_pred[n] = neg_binomial_2_rng(mu, phi);
//     }
//   }
// }
