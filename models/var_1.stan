data {
    int<lower=1> T;
    int<lower=1> n_eqs;
    int<lower=1> n_lags;
    int<lower=1> n_cross_vars;
    int<lower=0, upper=1> isChol;

    real lag_coef_mu;
    real<lower=0> lag_coef_sigma;

    real alpha_mu;
    real<lower=0> alpha_sigma;

    real noise_chol_eta;
    real<lower=0> noise_chol_sigma;

    real<lower=0> noise_sigma;
    
    matrix[T, n_eqs] y;
    
}

transformed data {
    vector<lower=0>[n_eqs] chol_sigma;
    for (i in 1:n_eqs) {
        chol_sigma[i] = noise_chol_sigma * 1;
    }
}

parameters {
    array[n_eqs] matrix[n_lags, n_cross_vars] lag_coefs;
    vector[n_eqs] alpha;
    cholesky_factor_corr[n_eqs] L_corr;
}

transformed parameters {
    matrix[T, n_eqs] mean; 
    matrix[T, n_eqs] beta; 

    for (j in 1:n_eqs) {
        for (t in n_lags+1:T) {
            beta[t, j] = 0;
            for (i in 1:n_lags) {
                for (k in 1:n_cross_vars) {
                    beta[t, j] += lag_coefs[j][i, k] * y[t-i, k];
                }
            }
            mean[t, j] = alpha[j] + beta[t, j];
        }
    }

}

model {
    for (j in 1:n_eqs) {
        for (i in 1:n_lags) {
            for (k in 1:n_cross_vars) {
                lag_coefs[j, i, k] ~ normal(lag_coef_mu, lag_coef_sigma);
            }
        }
    }

    alpha ~ normal(alpha_mu, alpha_sigma);
    L_corr ~ lkj_corr_cholesky(noise_chol_eta);

    if (isChol) {
        for (t in 1+n_lags:T) {
                y[t, ] ~ multi_normal_cholesky(mean[t, ], diag_pre_multiply(chol_sigma, L_corr));
        }
    }
}

generated quantities {
    matrix[T, n_eqs] y_pred;

    for (t in 1+n_lags:T) {
        y_pred[t, ] = to_row_vector(multi_normal_cholesky_rng(mean[t, ], diag_pre_multiply(chol_sigma, L_corr)));
    }
}
