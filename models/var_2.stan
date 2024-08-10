data {
    int<lower=1> T; // Total Number of Observations
    int<lower=1> n_groups; // Number of Countries Values Being Integers 0-7
    int<lower=1> n_eqs;
    int<lower=1> n_lags;
    int<lower=1> n_cross_vars;

    // Hierarchical Priors
    real<lower=0> rho_alpha;
    real<lower=0> rho_beta;

    real<lower=0> alpha_hat_location_mu;
    real<lower=0> alpha_hat_location_sigma;

    real<lower=0> alpha_hat_scale_mu;
    real<lower=0> alpha_hat_scale_sigma;

    real<lower=0> beta_hat_location_mu;
    real<lower=0> beta_hat_location_sigma;

    real<lower=0> beta_hat_scale_mu;
    real<lower=0> beta_hat_scale_sigma;

    real<lower=0> omega_eta;
    real<lower=0> omega_sigma_exponential;

    // Priors
    real<lower=0> z_scale_beta_mu;
    real<lower=0> z_scale_beta_sigma;

    real<lower=0> z_scale_alpha_mu;
    real<lower=0> z_scale_alpha_sigma;

    real<lower=0> noise_chol_eta;
    real<lower=0> noise_chol_sigma_exponential;

    // Data
    array[n_groups] matrix[T, n_eqs] Y;
}

transformed data {
    vector<lower=0>[n_eqs] omega_sigma_exponential_vector;
    for (i in 1:n_eqs) {
        omega_sigma_exponential_vector[i] = omega_sigma_exponential * 1;
    }

    vector<lower=0>[n_eqs] noise_chol_sigma_exponential_vector;
    for (i in 1:n_eqs) {
        noise_chol_sigma_exponential_vector[i] = noise_chol_sigma_exponential * 1;
    }
}

parameters {
    real<lower=0, upper=1> rho;
    real alpha_hat_location;
    real<lower=0> alpha_hat_scale;
    real beta_hat_location;
    real<lower=0> beta_hat_scale;

    cholesky_factor_corr[n_eqs] L_Omega_global;

    vector<lower=0>[n_eqs] omega_sigma;
    vector<lower=0>[n_eqs] noise_chol_sigma;

    array[n_groups] vector[n_eqs] alpha;

    array[n_groups, n_eqs] matrix[n_lags, n_cross_vars] lag_coefs;

    array[n_groups] real<lower=0> z_scale_alpha;
    array[n_groups] real<lower=0> z_scale_beta;

    array[n_groups] cholesky_factor_corr[n_eqs] L_noise_corr;
}

transformed parameters {
    array[n_groups] matrix[T, n_eqs] mean; 
    array[n_groups] matrix[T, n_eqs] beta; 

    for (g in 1:n_groups) {
        for (j in 1:n_eqs) {
            for (t in (n_lags + 1):T) {
                beta[g][t, j] = 0;
                for (i in 1:n_lags) {
                    for (k in 1:n_cross_vars) {
                        beta[g][t, j] += lag_coefs[g, j][i, k] * Y[g][t - i, k];
                    }
                }
                mean[g][t, j] = alpha[g, j] + beta[g][t, j];
            }
        }
    }
}

model {
    rho ~ beta(rho_alpha, rho_beta);
    alpha_hat_location ~ normal(alpha_hat_location_mu, alpha_hat_location_sigma);
    alpha_hat_scale ~ inv_gamma(alpha_hat_scale_mu, alpha_hat_scale_sigma);
    beta_hat_location ~ normal(beta_hat_location_mu, beta_hat_location_sigma);
    beta_hat_scale ~ inv_gamma(beta_hat_scale_mu, beta_hat_scale_sigma);
    L_Omega_global ~ lkj_corr_cholesky(omega_eta);
    omega_sigma ~ exponential(omega_sigma_exponential_vector);
    noise_chol_sigma ~ exponential(noise_chol_sigma_exponential_vector);

    for (g in 1:n_groups) {
        for (j in 1:n_eqs) {
            for (i in 1:n_lags) {
                for (k in 1:n_cross_vars) {
                    lag_coefs[g, j, i, k] ~ normal(beta_hat_location, beta_hat_scale * z_scale_beta[g]);
                }
            }
        }
    }

    for (g in 1:n_groups) {
        alpha[g] ~ normal(alpha_hat_location, alpha_hat_scale * z_scale_alpha[g]);
    }

    for (g in 1:n_groups) {
        L_noise_corr[g] ~ lkj_corr_cholesky(noise_chol_eta);
    }

    for (g in 1:n_groups) {  
        for (t in 1+n_lags:T) {
            Y[g][t,] ~ multi_normal_cholesky(mean[g][t, ],
                                             rho * diag_pre_multiply(omega_sigma, L_Omega_global) +
                                                (1- rho) * diag_pre_multiply(noise_chol_sigma, L_noise_corr[g])
                                            );
        }
    }
}


generated quantities {
    array[n_groups] matrix[T, n_eqs] Y_rep; // Posterior predictive distribution
    
    for (g in 1:n_groups) {
        for (t in (n_lags + 1):T) {
            Y_rep[g][t, ] = to_row_vector(multi_normal_cholesky_rng(
                                mean[g][t, ],
                                rho * diag_pre_multiply(omega_sigma, L_Omega_global) +
                                (1 - rho) * diag_pre_multiply(noise_chol_sigma, L_noise_corr[g])
                            )); 
        }
    }
}


