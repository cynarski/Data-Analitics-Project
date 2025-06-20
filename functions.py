import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import poisson, nbinom
import xarray as xr


def scatter_hist_display(val_mat, y_values):
    fig, axs = plt.subplots(2, len(val_mat.columns), dpi=300)
    fig.set_size_inches(5*len(val_mat.columns), 10)

    for i, col in enumerate(val_mat.columns):
        axs[0, i].scatter(val_mat[col], y_values)
        axs[0, i].set_title(col)
        axs[0, i].set_ylabel(y_values.name)
        axs[0, i].grid()

        axs[1, i].hist(val_mat[col])
        axs[1, i].set_ylabel("Density")
        axs[1, i].set_xlabel(col)
        axs[1, i].grid()

# def hist_compare(y_real, y_gen):
#     plt.figure(dpi=150)
#     y_real.hist(alpha=0.5, label="Real")
#     y_gen.hist(alpha=0.5, label="Generated", bins=23)
#     plt.legend()
#     plt.title("Generated samples")
#     plt.xlabel(y_real.name)
#     plt.grid()
#     plt.show()


def hist_compare(y_real, y_gen):
    plt.figure(dpi=150)
    max_val = int(max(y_real.max(), y_gen.max())) + 1
    bins = range(0, max_val + 1)
    y_real.hist(alpha=0.5, label="Real", bins=bins)
    y_gen.hist(alpha=0.5, label="Generated", bins=bins)
    plt.legend()
    plt.title("Generated samples")
    plt.xlabel(y_real.name)
    plt.xlim(0,20)
    plt.grid()
    plt.show()


def scatter_compare_display(val_mat, y_values_orig, y_values_gen):
    # plt.figure(dpi=300)
    fig, axs = plt.subplots(1, len(val_mat.columns), dpi=300)
    fig.set_size_inches(5 * len(val_mat.columns), 5)

    for i, col in enumerate(val_mat.columns):
        print(f"Length x (val_mat[col]): {len(val_mat[col])}")
        print(f"Length y (y_values_gen): {len(y_values_gen)}")
        axs[i].scatter(val_mat[col], y_values_orig, alpha=0.8, label="Original")
        axs[i].scatter(val_mat[col], y_values_gen, alpha=0.8, label="Generated")
        axs[i].set_title(col)
        axs[i].set_ylabel(y_values_orig.name)
        axs[i].set_xlabel(col)
        axs[i].grid()
        axs[i].legend()


def compute_poisson_log_likelihood(inference_data, X, nkill):
    """
    Compute pointwise log-likelihood for Poisson regression model
    """
    # Extract posterior samples
    alpha_samples = inference_data.posterior.alpha.values.flatten()
    beta_samples = inference_data.posterior.beta.values.reshape(-1, X.shape[1])

    n_samples = len(alpha_samples)
    n_obs = len(nkill)

    # Initialize log-likelihood array
    log_lik = np.zeros((n_samples, n_obs))

    # Compute log-likelihood for each sample and observation
    for i in range(n_samples):
        # Compute linear predictor
        eta = alpha_samples[i] + X @ beta_samples[i]
        # Compute lambda (rate parameter) - clip to prevent overflow
        lambda_i = np.exp(np.clip(eta, -500, 500))
        # Compute log-likelihood
        log_lik[i, :] = poisson.logpmf(nkill, lambda_i)

    return log_lik


def compute_negbinom_log_likelihood(inference_data, X, nkill):
    """
    Compute pointwise log-likelihood for Negative Binomial regression model
    """
    # Extract posterior samples
    alpha_samples = inference_data.posterior.alpha.values.flatten()
    beta_samples = inference_data.posterior.beta.values.reshape(-1, X.shape[1])
    phi_samples = inference_data.posterior.phi.values.flatten()

    n_samples = len(alpha_samples)
    n_obs = len(nkill)

    # Initialize log-likelihood array
    log_lik = np.zeros((n_samples, n_obs))

    # Compute log-likelihood for each sample and observation
    for i in range(n_samples):
        # Compute linear predictor
        eta = alpha_samples[i] + X @ beta_samples[i]
        # Compute mu (mean parameter) - clip to prevent overflow
        mu = np.exp(np.clip(eta, -500, 500))
        # Stan's neg_binomial_2 parameterization: n = phi, p = phi/(mu + phi)
        phi_i = phi_samples[i]
        p = phi_i / (mu + phi_i)
        # Compute log-likelihood using scipy's nbinom
        log_lik[i, :] = nbinom.logpmf(nkill, phi_i, p)

    return log_lik
